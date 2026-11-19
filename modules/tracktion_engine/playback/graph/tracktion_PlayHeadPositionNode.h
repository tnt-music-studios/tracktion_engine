/*
    ,--.                     ,--.     ,--.  ,--.
  ,-'  '-.,--.--.,--,--.,---.|  |,-.,-'  '-.`--' ,---. ,--,--,      Copyright 2024
  '-.  .-'|  .--' ,-.  | .--'|     /'-.  .-',--.| .-. ||      \   Tracktion Software
    |  |  |  |  \ '-'  \ `--.|  \  \  |  |  |  |' '-' '|  ||  |       Corporation
    `---' `--'   `--`--'`---'`--'`--' `---' `--' `---' `--''--'    www.tracktion.com

    Tracktion Engine uses a GPL/commercial licence - see LICENCE.md for details.
*/

namespace tracktion::inline engine {

/**
    A Node that calculates a position to show visually what time is currently being processed by the graph based on its internal latency.
*/
class PlayHeadPositionNode final    : public tracktion::graph::Node,
                                      public TracktionEngineNode
{
public:
    PlayHeadPositionNode (ProcessState& processStateToUse, std::unique_ptr<tracktion::graph::Node> inputNode,
                          std::atomic<double>& playHeadTimeToUpdate,
                          crill::seqlock_object<EditPlaybackContext::AudibleTime>& audibleTimeToUpdate)
        : TracktionEngineNode (processStateToUse),
          input (std::move (inputNode)),
          playHeadTime (playHeadTimeToUpdate),
          audibleTime (audibleTimeToUpdate)
    {
        setOptimisations ({ ClearBuffers::no,
                            AllocateAudioBuffer::no });
    }

    tracktion::graph::NodeProperties getNodeProperties() override
    {
        auto props = input->getNodeProperties();

        constexpr size_t playHeadPositionNodeMagicHash = size_t (0x706c617948656164);

        if (props.nodeID != 0)
            hash_combine (props.nodeID, playHeadPositionNodeMagicHash);

        return props;
    }

    std::vector<tracktion::graph::Node*> getDirectInputNodes() override  { return { input.get() }; }
    bool isReadyToProcess() override                                    { return input->hasProcessed(); }

    void prepareToPlay (const tracktion::graph::PlaybackInitialisationInfo& info) override
    {
        latencyNumSamples = info.nodeGraph.rootNode->getNodeProperties().latencyNumSamples;

        // Member variables have to be updated from the previous Node or if the graph gets
        // rebuilt during the countdown period, the playhead time will jump back
        updateFromPreviousNode (info.nodeGraphToReplace);
    }

    void process (ProcessContext& pc) override
    {
        // Copy the input buffers to the outputs
        auto sourceBuffers = input->getProcessedOutput();
        jassert (sourceBuffers.audio.getNumChannels() == pc.buffers.audio.getNumChannels());

        pc.buffers.midi.copyFrom (sourceBuffers.midi);
        setAudioOutput (input.get(), sourceBuffers.audio);
        updatePlayHeadTime (pc.referenceSampleRange.getLength());
    }

private:
    std::unique_ptr<tracktion::graph::Node> input;
    std::atomic<double>& playHeadTime;
    crill::seqlock_object<EditPlaybackContext::AudibleTime>& audibleTime;

    int latencyNumSamples = 0;
    bool updateReferencePositionOnJump = true;

    struct State
    {
        int64_t numLatencySamplesToCountDown = 0;
        int64_t referencePositionOnJump = 0;
    };

    std::shared_ptr<State> state { std::make_shared<State>() };

    void updatePlayHeadTime (int64_t numSamples)
    {
        const int64_t pieceReferenceStart = getReferenceSampleRange().getStart();
        int64_t referenceSamplePosition = pieceReferenceStart;

        if (getPlayHeadState().didPlayheadJump() || updateReferencePositionOnJump)
        {
            state->numLatencySamplesToCountDown = latencyNumSamples;
            state->referencePositionOnJump = referenceSamplePosition;
        }

        if (state->numLatencySamplesToCountDown > 0)
        {
            const int64_t numSamplesToDecrement = std::min (state->numLatencySamplesToCountDown, numSamples);
            state->numLatencySamplesToCountDown -= numSamplesToDecrement;
        }

        // Read once: the message thread can stop the playhead mid-call, and the published
        // record's flag has to agree with the hold chosen here.
        const bool advancing = getPlayHead().isPlaying() && state->numLatencySamplesToCountDown <= 0;

        if (advancing)
        {
            referenceSamplePosition -= latencyNumSamples;
        }
        else
        {
            referenceSamplePosition = state->referencePositionOnJump;
        }

        const int64_t timelinePosition = getPlayHead().referenceSamplePositionToTimelinePosition (referenceSamplePosition);
        const int64_t unloopedPosition = getPlayHead().referenceSamplePositionToTimelinePositionUnlooped (referenceSamplePosition);
        const double time = tracktion::graph::sampleToTime (timelinePosition, getSampleRate());

        playHeadTime = time;
        publishAudibleTime (time, unloopedPosition, pieceReferenceStart, numSamples, advancing);
        updateReferencePositionOnJump = false;
    }

    // Publishes this piece's audible time with the instant its first sample is heard. The last
    // published record is the predecessor -- the context's seed before the first piece -- so a
    // rebuilt node judges its first piece against it too. A piece continues the record when it
    // starts where that one ended, or where it held, on the play head's unlooped axis; anything
    // else (a seek, a tempo remap, a stop's replay, a device restart, a latency pin after a
    // rebuild) is counted, so a reader that skips pieces still sees that the timeline jumped.
    // A loop wrap continues the record: the unlooped axis runs on through it. A loop change
    // re-seats the play head and is counted when that moves the unlooped position. The loop is
    // read beside the position, not with it; the play head changes a loop in three
    // unsynchronized stores, so a piece rendered across a change may carry a mixed record, and
    // the next piece corrects it.
    void publishAudibleTime (double time, int64_t unloopedPosition, int64_t pieceReferenceStart,
                             int64_t numSamples, bool advancing)
    {
        const auto& processState = getProcessState();
        const double sampleRate = getSampleRate();
        const auto toSteadyDuration = [sampleRate] (int64_t samples)
        {
            return std::chrono::duration_cast<std::chrono::steady_clock::duration> (
                std::chrono::duration<double> (double (samples) / sampleRate));
        };

        const auto heardAt = processState.heardAtBlockStart
                              + toSteadyDuration (pieceReferenceStart - processState.blockReferenceStart);

        // This node is the record's only writer, so the load reads its own last store and never retries.
        const auto previous = audibleTime.load();
        // Both values were whole samples when stored, so the round trip is exact.
        const int64_t previousEnd = toSamples (previous.linearTime, sampleRate)
                                     + (previous.advancing ? toSamples (TimeDuration (previous.extent), sampleRate) : 0);
        const bool discontinuous = unloopedPosition != previousEnd;

        const auto runHeardFrom = (advancing && (! previous.advancing || discontinuous)) ? heardAt
                                                                                          : previous.runHeardFrom;

        const auto& playHead = getPlayHead();
        const auto loopRange = playHead.getLoopRange();
        const std::optional<TimeRange> loop = (playHead.isLooping() && ! playHead.isRollingIntoLoop() && ! loopRange.isEmpty())
                                                  ? std::optional<TimeRange> (timeRangeFromSamples (loopRange, sampleRate))
                                                  : std::nullopt;

        audibleTime.store ({ TimePosition::fromSeconds (time), heardAt, toSteadyDuration (numSamples), runHeardFrom,
                             advancing, previous.discontinuities + (discontinuous ? 1u : 0u),
                             TimePosition::fromSeconds (tracktion::graph::sampleToTime (unloopedPosition, sampleRate)), loop });
    }

    void updateFromPreviousNode (NodeGraph* nodeGraphToReplace)
    {
        if (auto oldNode = findNodeWithIDIfNonZero<PlayHeadPositionNode> (nodeGraphToReplace, getNodeProperties().nodeID))
        {
            state = oldNode->state;
            updateReferencePositionOnJump = false;
        }
    }
};

} // namespace tracktion::inline engine
