package com.comet.opik.api.resources.v1.events;

import com.comet.opik.api.FeedbackScoreItem;
import com.comet.opik.api.Trace;
import com.comet.opik.api.attachment.AttachmentInfo;
import com.comet.opik.api.evaluators.AutomationRuleEvaluatorType;
import com.comet.opik.api.events.WorkspaceScopedMessage;
import com.comet.opik.api.filter.Operator;
import com.comet.opik.api.filter.TraceField;
import com.comet.opik.api.filter.TraceFilter;
import com.comet.opik.api.resources.v1.events.tools.TraceToolContext;
import com.comet.opik.domain.FeedbackScoreService;
import com.comet.opik.domain.TraceSearchCriteria;
import com.comet.opik.domain.TraceService;
import com.comet.opik.domain.attachment.AttachmentUtils;
import com.comet.opik.infrastructure.OnlineScoringConfig;
import com.comet.opik.infrastructure.OnlineScoringStreamConfigurationAdapter;
import com.comet.opik.infrastructure.auth.RequestContext;
import com.comet.opik.utils.JsonUtils;
import com.fasterxml.jackson.databind.JsonNode;
import io.opentelemetry.api.GlobalOpenTelemetry;
import io.opentelemetry.api.common.AttributeKey;
import io.opentelemetry.api.common.Attributes;
import io.opentelemetry.api.metrics.LongCounter;
import jakarta.validation.constraints.NotNull;
import lombok.NonNull;
import org.apache.commons.lang3.StringUtils;
import org.redisson.api.RedissonReactiveClient;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import reactor.core.publisher.Flux;
import reactor.core.publisher.Mono;
import ru.vyarus.dropwizard.guice.module.yaml.bind.Config;

import java.math.BigDecimal;
import java.time.Duration;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import java.util.UUID;
import java.util.concurrent.atomic.AtomicReference;
import java.util.stream.Collectors;

import static com.comet.opik.api.FeedbackScoreItem.FeedbackScoreBatchItem;
import static com.comet.opik.api.FeedbackScoreItem.FeedbackScoreBatchItemThread;
import static com.comet.opik.infrastructure.log.LogContextAware.wrapWithMdc;

/**
 * Base online scorer for all particular implementations to extend. It listens to a Redis stream for
 * Traces/Spans/Threads to be scored. Subclasses provide a particular {@link #score(Object)} implementation that
 * returns a {@link Mono} so the entire processing chain stays non-blocking from Redis read to feedback-score
 * persistence. The Reactor pipeline owned by {@link BaseRedisSubscriber} schedules execution on the per-stream
 * worker scheduler; subclasses should NOT call {@code .block()} from {@code score()}.
 */
public abstract class OnlineScoringBaseScorer<M extends WorkspaceScopedMessage> extends BaseRedisSubscriber<M> {

    public static final int TRACE_PAGE_LIMIT = 2000;

    /**
     * Attachment-upload race tolerance for the {@code {{trace}}} / {@code {{span}}} structures. The SDK
     * uploads an entity's attachment a short moment <em>after</em> the entity itself is ingested, so a
     * scoring run triggered immediately can read the attachment table before the persistent copy lands.
     * When the entity body references an attachment but the listing is still empty, the cold lookup is
     * resubscribed up to {@link #ATTACHMENT_FETCH_MAX_RETRIES} times spaced by
     * {@link #ATTACHMENT_FETCH_RETRY_DELAY} so the upload can complete (~0.3–1.5 s worst case, and only
     * for entities that actually expect an attachment). See {@link #listAttachmentsToleratingUploadRace}.
     */
    private static final int ATTACHMENT_FETCH_MAX_RETRIES = 5;
    private static final Duration ATTACHMENT_FETCH_RETRY_DELAY = Duration.ofMillis(300);

    private static final String ONLINE_SCORING_NAMESPACE = "online_scoring";
    private static final AttributeKey<String> WORKSPACE_ID_KEY = AttributeKey.stringKey("workspace_id");
    private static final AttributeKey<String> WORKSPACE_NAME_KEY = AttributeKey.stringKey("workspace_name");

    /**
     * Logger for the actual subclass, in order to have the correct class name in the logs.
     */
    private final Logger log = LoggerFactory.getLogger(this.getClass());

    protected final FeedbackScoreService feedbackScoreService;
    protected final TraceService traceService;
    protected final AutomationRuleEvaluatorType type;

    /**
     * Per-workspace count of messages successfully processed (scored) by this stream. Together with
     * {@code online_scoring_<scorer>_processing_errors_total} (failures) this gives the consumer-side
     * processed-vs-failed split per workspace. Exported as {@code online_scoring_<scorer>_processed_total}.
     */
    private final LongCounter processedCounter;

    protected OnlineScoringBaseScorer(@NonNull @Config OnlineScoringConfig config,
            @NonNull RedissonReactiveClient redisson,
            @NonNull FeedbackScoreService feedbackScoreService,
            @NonNull TraceService traceService,
            @NonNull AutomationRuleEvaluatorType type,
            @NonNull String metricsBaseName) {
        super(OnlineScoringStreamConfigurationAdapter.create(config, type),
                redisson,
                OnlineScoringConfig.PAYLOAD_FIELD,
                ONLINE_SCORING_NAMESPACE,
                metricsBaseName);
        this.feedbackScoreService = feedbackScoreService;
        this.traceService = traceService;
        this.type = type;
        this.processedCounter = GlobalOpenTelemetry.getMeter(ONLINE_SCORING_NAMESPACE)
                .counterBuilder("%s_%s_processed".formatted(ONLINE_SCORING_NAMESPACE, metricsBaseName))
                .setDescription("Messages successfully processed (scored), by workspace")
                .build();
    }

    /**
     * Shared error surfacing for the agentic-tools path: when the tool-call loop fails after at
     * least one attachment was injected as multimodal content, the most likely cause is the judge
     * model rejecting that media type (we attempt all types rather than pre-gating). Emit a clear,
     * attachment-attributed user-facing message before propagating, so a vision-incapable model
     * produces an understandable error rather than a raw provider stack trace. With no injected
     * media the failure passes through untouched.
     *
     * <p>Static + parameterized on {@code userFacingLogger} / {@code modelName} so the trace-, span-
     * and thread-level scorers can all reuse it despite each owning its own logger and model accessor.
     */
    protected static <T> Mono<T> surfaceInjectedMediaFailure(@NonNull Throwable error,
            @NonNull TraceToolContext ctx, String modelName, @NonNull Logger userFacingLogger,
            @NonNull Map<String, String> mdc) {
        if (ctx.hasInjectedMedia()) {
            String attachments = ctx.getInjectedAttachments().stream()
                    .map(a -> "'%s' (%s)".formatted(a.fileName(), a.category().name().toLowerCase()))
                    .collect(Collectors.joining(", "));
            String detail = Optional.ofNullable(error.getCause()).map(Throwable::getMessage)
                    .orElse(error.getMessage());
            try (var logContext = wrapWithMdc(mdc)) {
                userFacingLogger.error(
                        "Scoring failed after loading attachment(s) {}; the judge model '{}' may not support this"
                                + " attachment type. Use a model that supports the attachment's media type. Details: {}",
                        attachments, modelName, detail);
            }
        }
        return Mono.error(error);
    }

    /**
     * Lists an entity's attachments while tolerating the upload race (see
     * {@link #ATTACHMENT_FETCH_MAX_RETRIES}). Shared by the trace- and span-level scorers when building
     * the injected {@code {{trace}}} / {@code {{span}}} structure.
     *
     * <p>An upload exists transiently as an <em>auto-stripped</em> copy ({@code input-attachment-N-ts.ext},
     * no {@code -sdk}) that is <strong>deleted</strong> once the persistent copy (e.g. {@code …-sdk.jpg})
     * lands. A listing taken mid-race can therefore contain only the soon-to-404 transient name. So when
     * any of {@code bodyNodes} (the entity's input/output/metadata) references an attachment, the cold
     * lookup is resubscribed a few times with a short delay until a <em>persistent</em> (non-auto-stripped)
     * attachment appears, and transient copies are dropped whenever a persistent one is present (so the
     * judge is never handed a name that will 404). Entities with no attachment reference skip the retry
     * (the common case). If the retry budget is exhausted — e.g. a REST-ingested image whose only copy is
     * auto-stripped and never replaced — it falls back to a best-effort final read rather than dropping it.
     *
     * @param coldFetch the attachment lookup — must be cold (re-runs the query on each subscription)
     * @param bodyNodes the entity's content nodes scanned for attachment references
     */
    protected static Mono<List<AttachmentInfo>> listAttachmentsToleratingUploadRace(
            @NonNull Mono<List<AttachmentInfo>> coldFetch, JsonNode... bodyNodes) {
        boolean expectsAttachment = AttachmentUtils.hasAttachmentReferences(JsonUtils.getMapper(), bodyNodes);
        if (!expectsAttachment) {
            return coldFetch.map(OnlineScoringBaseScorer::preferPersistentAttachments).onErrorReturn(List.of());
        }
        return coldFetch
                .map(OnlineScoringBaseScorer::preferPersistentAttachments)
                .filter(OnlineScoringBaseScorer::hasPersistentAttachment)
                .repeatWhenEmpty(ATTACHMENT_FETCH_MAX_RETRIES,
                        repeats -> repeats.delayElements(ATTACHMENT_FETCH_RETRY_DELAY))
                .onErrorResume(error -> Mono.empty())
                // Retries exhausted (no persistent copy will come): best-effort final read so a
                // backend-/REST-only auto-stripped attachment is still surfaced rather than dropped.
                .switchIfEmpty(Mono.defer(() -> coldFetch
                        .map(OnlineScoringBaseScorer::preferPersistentAttachments)
                        .onErrorReturn(List.of())));
    }

    /**
     * When both a transient auto-stripped copy and a persistent copy of an upload are present, drops the
     * auto-stripped one (it is deleted once the persistent copy lands, so surfacing it hands the judge a
     * name that 404s). When only auto-stripped copies exist (a backend-/REST-ingested image with no SDK
     * copy), they are the real attachments and are kept as-is.
     */
    private static List<AttachmentInfo> preferPersistentAttachments(List<AttachmentInfo> attachments) {
        if (!hasPersistentAttachment(attachments)) {
            return attachments;
        }
        return attachments.stream()
                .filter(attachment -> !AttachmentUtils.isAutoStrippedAttachment(attachment.fileName()))
                .collect(Collectors.toList());
    }

    private static boolean hasPersistentAttachment(List<AttachmentInfo> attachments) {
        return attachments.stream().anyMatch(a -> !AttachmentUtils.isAutoStrippedAttachment(a.fileName()));
    }

    /**
     * Defers the subscription of {@link #score(Object)} so any synchronous work in implementations
     * runs at subscription time on the per-stream worker scheduler.
     */
    @Override
    protected final Mono<Void> processEvent(M message) {
        var workspaceName = StringUtils.defaultIfBlank(message.workspaceName(), message.workspaceId());
        return doScore(message)
                .doOnSuccess(ignored -> processedCounter.add(1, Attributes.of(
                        WORKSPACE_ID_KEY, message.workspaceId(),
                        WORKSPACE_NAME_KEY, workspaceName)))
                // Carry both workspace id and name on the reactive context for the whole chain, sourced
                // from the message (resolved from RequestContext.WORKSPACE_NAME at trace-event publish time).
                .contextWrite(ctx -> ctx
                        .put(RequestContext.WORKSPACE_ID, message.workspaceId())
                        .put(RequestContext.WORKSPACE_NAME, workspaceName)
                        .put(RequestContext.USER_NAME, message.userName()));
    }

    /**
     * Full per-message processing chain. Defaults to {@link #score(Object)}, deferred so any
     * synchronous work runs at subscription time on the per-stream worker scheduler. Subclasses
     * that need post-scoring steps (e.g. test-suite assertion finalization) override this — not
     * {@code processEvent} — so the processed-success counter fires only once the whole chain
     * completes successfully.
     */
    protected Mono<Void> doScore(M message) {
        return Mono.defer(() -> score(message));
    }

    /**
     * Attributes processing-error metrics to the workspace/user the message belongs to. Without this
     * override the base class falls back to {@link MessageContext#UNKNOWN}, which is why
     * {@code online_scoring_*_processing_errors_total} historically reported {@code workspace_id="unknown"}.
     * The workspace name is carried on the message (resolved from RequestContext.WORKSPACE_NAME at
     * trace-event publish time); falls back to the id when absent.
     */
    @Override
    protected MessageContext messageContext(M message) {
        return MessageContext.builder()
                .workspaceId(message.workspaceId())
                .workspaceName(StringUtils.defaultIfBlank(message.workspaceName(), message.workspaceId()))
                .userName(message.userName())
                .build();
    }

    /**
     * Scores the message and persists the resulting feedback scores. Implementations must compose
     * reactive operators (no {@code .block()}); see {@link #storeScores}, {@link #storeSpanScores},
     * {@link #storeThreadScores}.
     */
    protected abstract Mono<Void> score(M message);

    protected Mono<Map<String, List<BigDecimal>>> storeScores(
            List<FeedbackScoreBatchItem> scores, Trace trace, String userName, String workspaceId) {
        log.info("Received '{}' scores for traceId '{}' in workspace '{}'. Storing them",
                scores.size(), trace.id(), workspaceId);
        return feedbackScoreService.scoreBatchOfTraces(scores)
                .contextWrite(ctx -> ctx.put(RequestContext.USER_NAME, userName)
                        .put(RequestContext.WORKSPACE_ID, workspaceId))
                .thenReturn(groupScoresByName(scores));
    }

    protected Mono<Map<String, List<BigDecimal>>> storeSpanScores(
            List<FeedbackScoreBatchItem> scores, com.comet.opik.api.Span span, String userName, String workspaceId) {
        log.info("Received '{}' scores for spanId '{}' in workspace '{}'. Storing them",
                scores.size(), span.id(), workspaceId);
        return feedbackScoreService.scoreBatchOfSpans(scores)
                .contextWrite(ctx -> ctx.put(RequestContext.USER_NAME, userName)
                        .put(RequestContext.WORKSPACE_ID, workspaceId))
                .thenReturn(groupScoresByName(scores));
    }

    protected Mono<Map<String, List<BigDecimal>>> storeThreadScores(
            List<FeedbackScoreBatchItemThread> scores, String threadId, String userName, String workspaceId) {
        log.info("Received '{}' scores for threadId '{}' in workspace '{}'. Storing them",
                scores.size(), threadId, workspaceId);
        return feedbackScoreService.scoreBatchOfThreads(scores)
                .contextWrite(ctx -> ctx.put(RequestContext.USER_NAME, userName)
                        .put(RequestContext.WORKSPACE_ID, workspaceId))
                .thenReturn(groupScoresByName(scores));
    }

    private static <T extends FeedbackScoreItem> Map<String, List<BigDecimal>> groupScoresByName(List<T> scores) {
        return scores.stream()
                .collect(Collectors.groupingBy(FeedbackScoreItem::name,
                        Collectors.mapping(FeedbackScoreItem::value, Collectors.toList())));
    }

    /**
     * Retrieves the full thread context for a given thread ID, recursively fetching traces until no more are found.
     *
     * @param threadId the ID of the thread to retrieve context for
     * @param lastReceivedIdRef a reference to store the last received trace ID
     * @param projectId the ID of the project to which the thread belongs
     * @return a Flux of Trace objects representing the full thread context
     */
    //TODO: Move this to a common service or utility class
    protected Flux<Trace> retrieveFullThreadContext(@NotNull String threadId,
            @NotNull AtomicReference<UUID> lastReceivedIdRef, @NotNull UUID projectId) {

        return Flux.defer(() -> traceService.search(TRACE_PAGE_LIMIT, TraceSearchCriteria.builder()
                .projectId(projectId)
                .filters(List.of(TraceFilter.builder()
                        .field(TraceField.THREAD_ID)
                        .operator(Operator.EQUAL)
                        .value(threadId)
                        .build()))
                .lastReceivedId(lastReceivedIdRef.get())
                .build())
                .collectList()
                .flatMapMany(results -> {
                    if (results.isEmpty()) {
                        return Flux.empty();
                    }
                    lastReceivedIdRef.set(results.getLast().id());
                    return Flux.fromIterable(results)
                            .concatWith(Flux
                                    .defer(() -> retrieveFullThreadContext(threadId, lastReceivedIdRef, projectId)));
                }));
    }
}
