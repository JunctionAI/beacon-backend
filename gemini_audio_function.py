def generate_gemini_multispeaker_audio(script: str, speakers: List[Dict[str, str]] = None) -> bytes:
    """
    Generate audio from podcast script using Gemini-TTS multi-speaker dialogue

    Args:
        script: Podcast script with speaker labels (e.g., "Alex: Hello!\nLuna: Hi there!")
        speakers: List of speaker configs, e.g., [{"alias": "Alex", "voice": "Alnilam"}, ...]

    Returns:
        Audio bytes (MP3 format) or empty bytes on error
    """
    if not gemini_tts_client:
        print("⚠️  No Gemini TTS client - falling back to standard TTS")
        return b""

    try:
        # Default speakers: Alex (Alnilam - male) and Luna (Autonoe - female)
        if not speakers or len(speakers) < 2:
            speakers = [
                {"alias": "Alex", "voice": "Alnilam"},   # Curious & thoughtful explorer (male)
                {"alias": "Luna", "voice": "Autonoe"}     # Enthusiastic & energetic co-host (female)
            ]

        print(f"🎙️ Using Gemini-TTS multi-speaker with {len(speakers)} voices")
        for speaker in speakers:
            print(f"   {speaker['alias']}: {speaker['voice']}")

        # Gemini-TTS has a 4000 byte limit per request
        # Split long scripts into chunks while preserving speaker turns
        max_chunk_size = 3500  # Leave buffer for encoding

        # Split script into lines
        lines = script.strip().split('\n')

        # Group into chunks
        chunks = []
        current_chunk = []
        current_size = 0

        for line in lines:
            line = line.strip()
            if not line:
                continue

            line_size = len(line.encode('utf-8'))

            # If adding this line exceeds limit, start new chunk
            if current_size + line_size > max_chunk_size and current_chunk:
                chunks.append('\n'.join(current_chunk))
                current_chunk = [line]
                current_size = line_size
            else:
                current_chunk.append(line)
                current_size += line_size

        # Add final chunk
        if current_chunk:
            chunks.append('\n'.join(current_chunk))

        print(f"📝 Split script into {len(chunks)} chunk(s) for Gemini-TTS")

        # Generate audio for each chunk
        audio_chunks = []

        for chunk_idx, chunk_text in enumerate(chunks):
            print(f"🎙️ Generating chunk {chunk_idx + 1}/{len(chunks)}...")
            print(f"   Preview: {chunk_text[:100]}...")

            try:
                # Configure multi-speaker voices
                multi_speaker_config = MultiSpeakerVoiceConfig(
                    speaker_voice_configs=[
                        MultispeakerPrebuiltVoice(
                            speaker_alias=speaker["alias"],
                            speaker_id=speaker["voice"]
                        ) for speaker in speakers
                    ]
                )

                # Synthesis input with speaker-labeled dialogue
                synthesis_input = GeminiSynthesisInput(text=chunk_text)

                # Audio configuration
                audio_config = GeminiAudioConfig(
                    audio_encoding=GeminiAudioEncoding.MP3,
                    effects_profile_id=["headphone-class-device"]
                )

                # Generate audio using Gemini-TTS multi-speaker dialogue API
                response = gemini_tts_client.synthesize_speech(
                    input=synthesis_input,
                    voice=GeminiVoiceParams(
                        language_code="en-US",
                        model_name="gemini-2.5-flash-tts",
                        multi_speaker_voice_config=multi_speaker_config
                    ),
                    audio_config=audio_config
                )

                audio_chunks.append(response.audio_content)
                print(f"   ✅ Generated {len(response.audio_content) / 1024:.1f} KB")

            except Exception as e:
                print(f"⚠️ Chunk {chunk_idx + 1} failed: {e}")
                import traceback
                print(traceback.format_exc())
                continue  # Skip failed chunk

        # Combine all audio chunks
        full_audio = b''.join(audio_chunks)

        print(f"✅ Generated {len(full_audio) / (1024*1024):.2f} MB of audio from {len(chunks)} chunk(s)")
        print(f"💰 Estimated cost: ${len(script) / 1000000 * 16:.2f} (@ $16 per 1M chars)")

        return full_audio

    except Exception as e:
        print(f"❌ Gemini-TTS audio generation error: {e}")
        import traceback
        print(traceback.format_exc())
        return b""
