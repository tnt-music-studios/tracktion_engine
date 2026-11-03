/*
    ,--.                     ,--.     ,--.  ,--.
  ,-'  '-.,--.--.,--,--.,---.|  |,-.,-'  '-.`--' ,---. ,--,--,      Copyright 2024
  '-.  .-'|  .--' ,-.  | .--'|     /'-.  .-',--.| .-. ||      \   Tracktion Software
    |  |  |  |  \ '-'  \ `--.|  \  \  |  |  |  |' '-' '|  ||  |       Corporation
    `---' `--'   `--`--'`---'`--'`--' `---' `--' `---' `--''--'    www.tracktion.com

    Tracktion Engine uses a GPL/commercial licence - see LICENCE.md for details.
*/

namespace tracktion::inline engine {

struct DbTimePair
{
    uint32_t time = 0;
    float dB = -100.0f;
};

//==============================================================================
/**
    Monitors the levels of buffers that are passed in, and keeps peak values,
    overloads, etc., for display in a level meter component.
*/
class LevelMeasurer
{
public:
    //==============================================================================
    LevelMeasurer();
    ~LevelMeasurer();

    //==============================================================================
    void processBuffer (juce::AudioBuffer<float>& buffer, int start, int numSamples);
    void processMidi (MidiMessageArray& midiBuffer, const float* gains);
    void processMidiLevel (float level);

    void clear();
    void clearOverload();
    void clearPeak();

    //==============================================================================
    enum Mode
    {
        peakMode     = 0,
        RMSMode      = 1,
        sumDiffMode  = 2
    };

    void setMode (Mode);
    Mode getMode() const noexcept                       { return mode; }

    void setShowMidi (bool showMidi);

    int getNumActiveChannels() const noexcept           { return numActiveChannels; }

    /** Receives levels from the audio thread and hands them to any other thread, without a lock
        and without allocating. A fixed number of channels makes that possible: 64 covers
        seventh-order ambisonics, and channels past it are not metered.
    */
    struct Client
    {
        static constexpr int maxChannels = 64;

        Client() = default;

        int getNumChannelsUsed() const noexcept;
        void reset() noexcept;
        bool getAndClearOverload() noexcept;
        bool getAndClearPeak() noexcept;
        DbTimePair getAndClearMidiLevel() noexcept;
        DbTimePair getAndClearAudioLevel (int chan) noexcept;

        /** @internal */
        void setNumChannelsUsed (int) noexcept;
        void setClearOverload (bool) noexcept;
        void setClearPeak (bool) noexcept;

        void updateAudioLevel (int channel, DbTimePair) noexcept;
        void updateMidiLevel (DbTimePair) noexcept;

        /** Updates the first numChannels levels and numChannelsUsed. */
        void updateAllChannels (const DbTimePair* levels, int numChannels) noexcept;

    private:
        static_assert (std::atomic<DbTimePair>::is_always_lock_free);

        std::array<std::atomic<DbTimePair>, (size_t) maxChannels> audioLevels;
        std::atomic<DbTimePair> midiLevels;
        std::atomic<int> numChannelsUsed { 0 };
        std::atomic<bool> clearOverload { true };
        std::atomic<bool> clearPeak { true };
    };

    //==============================================================================
    void addClient (Client&);
    void removeClient (Client&);

private:
    Mode mode = peakMode;
    std::atomic<int> numActiveChannels { 1 };
    bool showMidi = false;

    juce::Array<Client*> clients;
    RealTimeSpinLock clientsMutex;

    std::array<DbTimePair, (size_t) Client::maxChannels> channelLevels;

    JUCE_DECLARE_WEAK_REFERENCEABLE(LevelMeasurer)
    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR (LevelMeasurer)
};

//==============================================================================
/** A shared level measurer that can be used by several threads to provide a
    total output level
*/
class SharedLevelMeasurer  : public LevelMeasurer,
                             public juce::ReferenceCountedObject
{
public:
    using Ptr = juce::ReferenceCountedObjectPtr<SharedLevelMeasurer>;

    void startNextBlock (double streamTime);
    void setSize (int channels, int numSamples);
    void addBuffer (const juce::AudioBuffer<float>& inBuffer, int startSample, int numSamples);

    juce::SpinLock spinLock;
    double lastStreamTime = 0;
    juce::AudioBuffer<float> sumBuffer;
};

} // namespace tracktion::inline engine
