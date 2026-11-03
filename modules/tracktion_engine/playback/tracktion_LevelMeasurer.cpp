/*
    ,--.                     ,--.     ,--.  ,--.
  ,-'  '-.,--.--.,--,--.,---.|  |,-.,-'  '-.`--' ,---. ,--,--,      Copyright 2024
  '-.  .-'|  .--' ,-.  | .--'|     /'-.  .-',--.| .-. ||      \   Tracktion Software
    |  |  |  |  \ '-'  \ `--.|  \  \  |  |  |  |' '-' '|  ||  |       Corporation
    `---' `--'   `--`--'`---'`--'`--' `---' `--' `---' `--''--'    www.tracktion.com

    Tracktion Engine uses a GPL/commercial licence - see LICENCE.md for details.
*/

namespace tracktion::inline engine {

//==============================================================================
template <typename FloatType>
static void getSumAndDiff (const juce::AudioBuffer<FloatType>& buffer,
                           FloatType& sum, FloatType& diff,
                           int startIndex, int numSamples)
{
    if (buffer.getNumChannels() == 0)
    {
        sum = 0;
        diff = 0;
    }
    else
    {
        FloatType s  = 0;
        FloatType lo = (FloatType) 1;
        FloatType hi = 0;

        for (int i = buffer.getNumChannels(); --i >= 0;)
        {
            auto mag = buffer.getMagnitude (i, startIndex, numSamples);
            s += mag;
            lo = juce::jmin (lo, mag);
            hi = juce::jmax (hi, mag);
        }

        sum = s / static_cast<FloatType> (buffer.getNumChannels());
        diff = juce::jmax (FloatType(), hi - lo);
    }
}

//==============================================================================
LevelMeasurer::LevelMeasurer()
{
    clear();
}

LevelMeasurer::~LevelMeasurer()
{
    TRACKTION_ASSERT_MESSAGE_THREAD
}

//==============================================================================
int LevelMeasurer::Client::getNumChannelsUsed() const noexcept
{
    return numChannelsUsed;
}

void LevelMeasurer::Client::reset() noexcept
{
    for (auto& level : audioLevels)
        level = DbTimePair();

    midiLevels = DbTimePair();
    clearOverload = true;
}

bool LevelMeasurer::Client::getAndClearOverload() noexcept
{
    return clearOverload.exchange (false);
}

bool LevelMeasurer::Client::getAndClearPeak() noexcept
{
    return clearPeak.exchange (false);
}

DbTimePair LevelMeasurer::Client::getAndClearMidiLevel() noexcept
{
    return midiLevels.exchange (DbTimePair());
}

DbTimePair LevelMeasurer::Client::getAndClearAudioLevel (int chan) noexcept
{
    if (chan < 0 || chan >= maxChannels)
        return {};

    return audioLevels[(size_t) chan].exchange (DbTimePair());
}

void LevelMeasurer::Client::setNumChannelsUsed (int numChannels) noexcept
{
    numChannelsUsed = std::min (numChannels, maxChannels);
}

void LevelMeasurer::Client::setClearOverload (bool clear) noexcept
{
    clearOverload = clear;
}

void LevelMeasurer::Client::setClearPeak (bool clear) noexcept
{
    clearPeak = clear;
}

// Even a quieter level goes through the exchange: a reader clearing the slot since the load would
// otherwise lose this block's level, because the comparison was against a value already taken.
static void keepLouder (std::atomic<DbTimePair>& level, DbTimePair candidate) noexcept
{
    auto current = level.load();

    while (! level.compare_exchange_weak (current, candidate.dB >= current.dB ? candidate : current))
    {}
}

void LevelMeasurer::Client::updateAudioLevel (int channel, DbTimePair newAudioLevel) noexcept
{
    if (channel >= 0 && channel < maxChannels)
        keepLouder (audioLevels[(size_t) channel], newAudioLevel);
}

void LevelMeasurer::Client::updateMidiLevel (DbTimePair newMidiLevel) noexcept
{
    keepLouder (midiLevels, newMidiLevel);
}

void LevelMeasurer::Client::updateAllChannels (const DbTimePair* levels, int numChannels) noexcept
{
    setNumChannelsUsed (numChannels);

    for (int i = 0; i < numChannelsUsed; ++i)
        keepLouder (audioLevels[(size_t) i], levels[i]);
}

//==============================================================================
void LevelMeasurer::processBuffer (juce::AudioBuffer<float>& buffer, int start, int numSamples)
{
    const std::scoped_lock sl (clientsMutex);

    if (clients.isEmpty())
        return;

    auto numChans = buffer.getNumChannels();
    auto now = juce::Time::getApproximateMillisecondCounter();

    // Compute all channel levels first, then batch-update each client with a single lock
    int resultNumChans;

    if (mode == LevelMeasurer::sumDiffMode)
    {
        float sum, diff;
        getSumAndDiff (buffer, sum, diff, start, numSamples);

        resultNumChans = 2;

        channelLevels[0] = { now, gainToDb (sum) };
        channelLevels[1] = { now, gainToDb (diff) };
    }
    else
    {
        resultNumChans = std::min (numChans, Client::maxChannels);

        for (int i = 0; i < resultNumChans; ++i)
        {
            auto gain = (mode == LevelMeasurer::peakMode)
                            ? buffer.getMagnitude (i, start, numSamples)
                            : buffer.getRMSLevel (i, start, numSamples);

            channelLevels[(size_t) i] = { now, gainToDb (gain) };
        }
    }

    numActiveChannels = resultNumChans;

    for (auto c : clients)
        c->updateAllChannels (channelLevels.data(), resultNumChans);
}

void LevelMeasurer::processMidi (MidiMessageArray& midiBuffer, const float*)
{
    const std::scoped_lock sl (clientsMutex);

    if (clients.isEmpty() || ! showMidi)
        return;

    float max = 0.0f;

    for (auto& m : midiBuffer)
        if (m.isNoteOn())
            max = juce::jmax (max, m.getFloatVelocity());

    auto now = juce::Time::getApproximateMillisecondCounter();

    for (auto c : clients)
        c->updateMidiLevel ({ now, gainToDb (max) });
}

void LevelMeasurer::processMidiLevel (float level)
{
    const std::scoped_lock sl (clientsMutex);

    if (clients.isEmpty() || ! showMidi)
        return;

    auto now = juce::Time::getApproximateMillisecondCounter();

    for (auto c : clients)
        c->updateMidiLevel ({ now, gainToDb (level) });
}

void LevelMeasurer::clearOverload()
{
    const std::scoped_lock sl (clientsMutex);

    for (auto c : clients)
        c->setClearOverload (true);
}

void LevelMeasurer::clearPeak()
{
    const std::scoped_lock sl (clientsMutex);

    for (auto c : clients)
        c->setClearPeak (true);
}

void LevelMeasurer::clear()
{
    const std::scoped_lock sl (clientsMutex);

    for (auto c : clients)
        c->reset();
}

void LevelMeasurer::setMode (LevelMeasurer::Mode m)
{
    clear();
    mode = m;
}

void LevelMeasurer::addClient (Client& c)
{
    const std::scoped_lock sl (clientsMutex);
    jassert (! clients.contains (&c));
    clients.add (&c);
}

void LevelMeasurer::removeClient (Client& c)
{
    const std::scoped_lock sl (clientsMutex);
    clients.removeFirstMatchingValue (&c);
}

void LevelMeasurer::setShowMidi (bool show)
{
    showMidi = show;
}

//==============================================================================
void SharedLevelMeasurer::startNextBlock (double streamTime)
{
    juce::SpinLock::ScopedLockType lock (spinLock);

    if (streamTime != lastStreamTime)
    {
        lastStreamTime = streamTime;

        processBuffer (sumBuffer, 0, sumBuffer.getNumSamples());
        sumBuffer.clear();
    }
}

void SharedLevelMeasurer::setSize (int channels, int numSamples)
{
    juce::SpinLock::ScopedLockType lock (spinLock);

    if (channels > sumBuffer.getNumChannels() || numSamples > sumBuffer.getNumSamples())
        sumBuffer.setSize (channels, numSamples);
}

void SharedLevelMeasurer::addBuffer (const juce::AudioBuffer<float>& inBuffer, int startSample, int numSamples)
{
    setSize (inBuffer.getNumChannels(), numSamples);

    juce::SpinLock::ScopedLockType lock (spinLock);

    for (int i = 0; i < juce::jmin (sumBuffer.getNumChannels(), inBuffer.getNumChannels()); ++i)
        sumBuffer.addFrom (i, 0, inBuffer, i, startSample, numSamples);
}

} // namespace tracktion::inline engine
