import logging
from typing import List
from dotenv import load_dotenv
from datetime import datetime
from livekit.agents import (
    Agent,
    AgentSession,
    JobContext,
    JobProcess,
    MetricsCollectedEvent,
    RoomInputOptions,
    WorkerOptions,
    cli,
    metrics,
    tokenize,
    function_tool,
    RunContext,
)
from livekit.plugins import murf, silero, google, deepgram, noise_cancellation
from livekit.plugins.turn_detector.multilingual import MultilingualModel

logger = logging.getLogger("agent")

load_dotenv(".env.local")


class MiloAssistant(Agent):
    def __init__(self) -> None:
        super().__init__(
            instructions="""You are Milo, a personal appointment scheduling assistant.
            Your goal is to help users schedule their appointments.
            You need to collect the following information for each appointment: the user's intent or the purpose of the appointment, the date and time, the urgency level (e.g., high, medium, low) - the urgency would be default low unless specfied.
            You can summarize the users request before saving it.
            Once you have all this information, you must call the `add_appointment` tool to save it.
            You can also list all the scheduled appointments for the user if they ask for it, by using the `get_appointments` tool.
            You can also get the current date and time by using the `get_current_datetime` tool. This is useful for scheduling appointments relative to today.
            Start by greeting the user and asking how you can help them schedule something.""",
        )
        self.appointments: List[dict] = [] # You would probably have some kind of database, as of now its just inmemory

    @function_tool
    async def add_appointment(
        self,
        context: RunContext,
        intent: str,
        datetime: str,
        urgency: str,
        summary: str,
    ):
        """Use this tool to add a new appointment to the user's schedule.

        Args:
            intent: The purpose of the appointment.
            datetime: The date and time of the appointment.
            urgency: The urgency level of the appointment (e.g., high, medium, low).
            summary: A brief summary of the appointment.
        """

        logger.info(
            f"Adding appointment: intent='{intent}', datetime='{datetime}', urgency='{urgency}', summary='{summary}'"
        )

        appointment = {
            "intent": intent,
            "datetime": datetime,
            "urgency": urgency,
            "summary": summary,
        }
        self.appointments.append(appointment)

        return "I have successfully scheduled the appointment for you."

    @function_tool
    async def get_appointments(self, context: RunContext):
        """Use this tool to get the list of all scheduled appointments."""

        if not self.appointments:
            return "There are no appointments scheduled."

        formatted_appointments = "\n".join(
            [
                f"{i+1}. Intent: {a['intent']}, Date: {a['datetime']}, Urgency: {a['urgency']}, Summary: {a['summary']}"
                for i, a in enumerate(self.appointments)
            ]
        )

        return f"Here are your scheduled appointments:\n{formatted_appointments}"

    @function_tool
    async def get_current_datetime(self, context: RunContext):
        """Use this tool to get the current date and time."""
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def prewarm(proc: JobProcess):
    proc.userdata["vad"] = silero.VAD.load()


async def entrypoint(ctx: JobContext):
    # Logging setup
    # Add any other context you want in all log entries here
    ctx.log_context_fields = {
        "room": ctx.room.name,
    }

    # Set up a voice AI pipeline using OpenAI, Cartesia, AssemblyAI, and the LiveKit turn detector
    session = AgentSession(
        # Speech-to-text (STT) is your agent's ears, turning the user's speech into text that the LLM can understand
        # See all available models at https://docs.livekit.io/agents/models/stt/
        stt=deepgram.STT(model="nova-3"),
        # A Large Language Model (LLM) is your agent's brain, processing user input and generating a response
        # See all available models at https://docs.livekit.io/agents/models/llm/
        llm=google.LLM(
            model="gemini-2.5-flash",
        ),
        # Text-to-speech (TTS) is your agent's voice, turning the LLM's text into speech that the user can hear
        # See all available models as well as voice selections at https://docs.livekit.io/agents/models/tts/
        tts=murf.TTS(
            voice="en-US-matthew",
            style="Conversation",
            tokenizer=tokenize.basic.SentenceTokenizer(min_sentence_len=2),
            text_pacing=True,
            model="FALCON"
        ),
        # VAD and turn detection are used to determine when the user is speaking and when the agent should respond
        # See more at https://docs.livekit.io/agents/build/turns
        turn_detection=MultilingualModel(),
        vad=ctx.proc.userdata["vad"],
        # allow the LLM to generate a response while waiting for the end of turn
        # See more at https://docs.livekit.io/agents/build/audio/#preemptive-generation
        preemptive_generation=True,
    )

    # To use a realtime model instead of a voice pipeline, use the following session setup instead.
    # (Note: This is for the OpenAI Realtime API. For other providers, see https://docs.livekit.io/agents/models/realtime/))
    # 1. Install livekit-agents[openai]
    # 2. Set OPENAI_API_KEY in .env.local
    # 3. Add `from livekit.plugins import openai` to the top of this file
    # 4. Use the following session setup instead of the version above
    # session = AgentSession(
    #     llm=openai.realtime.RealtimeModel(voice="marin")
    # )

    # Metrics collection, to measure pipeline performance
    # For more information, see https://docs.livekit.io/agents/build/metrics/
    usage_collector = metrics.UsageCollector()

    @session.on("metrics_collected")
    def _on_metrics_collected(ev: MetricsCollectedEvent):
        metrics.log_metrics(ev.metrics)
        usage_collector.collect(ev.metrics)

    async def log_usage():
        summary = usage_collector.get_summary()
        logger.info(f"Usage: {summary}")

    ctx.add_shutdown_callback(log_usage)

    # # Add a virtual avatar to the session, if desired
    # # For other providers, see https://docs.livekit.io/agents/models/avatar/
    # avatar = hedra.AvatarSession(
    #   avatar_id="...",  # See https://docs.livekit.io/agents/models/avatar/plugins/hedra
    # )
    # # Start the avatar and wait for it to join
    # await avatar.start(session, room=ctx.room)

    # Start the session, which initializes the voice pipeline and warms up the models
    await session.start(
        agent=MiloAssistant(),
        room=ctx.room,
        room_input_options=RoomInputOptions(
            # For telephony applications, use `BVCTelephony` for best results
            noise_cancellation=noise_cancellation.BVC(),
        ),
    )

    # Join the room and connect to the user
    await ctx.connect()


if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint, prewarm_fnc=prewarm))
