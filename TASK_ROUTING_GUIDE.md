# Task Routing Guide: Intent-Based Agent Selection

## Overview

The video analyzer backend now supports **intent-based routing** - you can describe tasks in natural language and the system automatically routes them to the correct agent!

## How It Works

### 1. Agent Capabilities (Defined at top of each agent file)

Each agent declares its capabilities using the `AgentCapability` model:

**Example from [agents/transcription_agent.py](agents/transcription_agent.py):**
```python
TRANSCRIPTION_AGENT_CAPABILITIES = AgentCapability(
    capabilities=[
        "Audio transcription from video",
        "Speech-to-text conversion",
        "Subtitle generation",
    ],
    intent_keywords=[
        "transcribe", "transcript", "speech", "audio",
        "what said", "convert to text", "subtitle",
    ],
    categories=[CapabilityCategory.AUDIO, CapabilityCategory.TEXT],
    example_tasks=[
        "Transcribe the video",
        "What was said in the video?",
        "Create subtitles for the video",
    ],
    routing_priority=8,
)
```

**Example from [agents/vision_agent.py](agents/vision_agent.py):**
```python
VISION_AGENT_CAPABILITIES = AgentCapability(
    capabilities=[
        "Object detection in videos",
        "Visual content analysis",
        "People and animal detection",
    ],
    intent_keywords=[
        "detect", "identify", "find", "locate",
        "object", "person", "people", "car", "animal",
        "what see", "track", "movement",
    ],
    categories=[CapabilityCategory.VISION, CapabilityCategory.ANALYSIS],
    example_tasks=[
        "Detect objects in the video",
        "Find all people in the video",
        "What cars appear in the video?",
    ],
    routing_priority=9,
)
```

### 2. Intent Classifier

The `IntentClassifier` matches task descriptions against agent capabilities using keyword matching and scoring.

### 3. Automatic Routing

The `MultiAgentCoordinator` uses the classifier to route tasks automatically.

## Usage Examples

### Before (Old Way - Required exact task_type)

```python
# ❌ Had to specify exact task_type even when description was clear
VideoTask(
    description="Generate the transcript for the video",
    task_type="object_detection",  # Wrong! But required
    file_path="./sample.mp4"
)
```

### After (New Way - Just describe what you want!)

```python
# ✅ Just describe the task in natural language
VideoTask(
    description="Generate the transcript for the video",
    file_path="./sample.mp4"
)
# Automatically routes to: transcription_agent

VideoTask(
    description="Find all people in the video",
    file_path="./sample.mp4"
)
# Automatically routes to: vision_agent

VideoTask(
    description="What objects appear in this video?",
    file_path="./sample.mp4"
)
# Automatically routes to: vision_agent

VideoTask(
    description="Transcribe the audio from this video",
    file_path="./sample.mp4"
)
# Automatically routes to: transcription_agent
```

### Backward Compatibility

The old way still works! You can still specify `task_type` explicitly if needed:

```python
VideoTask(
    description="Analyze this video",
    task_type="object_detection",  # Explicit task type
    file_path="./sample.mp4"
)
```

## How to Add a New Agent

1. **Create capability definition at top of agent file:**

```python
# agents/new_agent.py

NEW_AGENT_CAPABILITIES = AgentCapability(
    capabilities=[
        "What your agent does",
        "Another capability",
    ],
    intent_keywords=[
        # Keywords that indicate this agent should handle the task
        "keyword1", "keyword2", "action phrase",
    ],
    categories=[CapabilityCategory.YOUR_CATEGORY],
    example_tasks=[
        "Example task description 1",
        "Example task description 2",
    ],
    routing_priority=5,  # 1-10, higher = preferred
)
```

2. **Use the capability in your agent class:**

```python
class NewAgent(BaseAgent):
    def __init__(self):
        super().__init__(
            name="new_agent",
            capabilities=NEW_AGENT_CAPABILITIES.capabilities
        )
        self.capability_definition = NEW_AGENT_CAPABILITIES

        # Register with the global registry
        from models.agent_capabilities import AgentCapabilityRegistry
        AgentCapabilityRegistry.register(self.name, self.capability_definition)

    def can_handle(self, task: Dict[str, Any]) -> bool:
        # Legacy support
        task_type = task.get("task_type", "").lower()
        if task_type in ["your_type"]:
            return True

        # New: Description-based matching
        description = task.get("description", "")
        if description:
            return self.capability_definition.matches_description(description)

        return False
```

3. **That's it!** The system automatically picks up your agent.

## Testing Routing

You can test what agent would be selected for a description:

```python
from routing.intent_classifier import get_intent_classifier

classifier = get_intent_classifier()

# Get the best agent
agent_name = classifier.get_best_agent("Transcribe the video")
print(agent_name)  # Output: transcription_agent

# Get all matching agents with scores
matches = classifier.classify("Find people and transcribe what they say")
print(matches)  # Output: [('vision_agent', 0.85), ('transcription_agent', 0.72)]

# Explain routing decision
explanation = classifier.explain_routing("Detect objects in the video")
print(explanation)
# Output: {
#     'description': 'Detect objects in the video',
#     'matches': [
#         {
#             'agent': 'vision_agent',
#             'score': 0.95,
#             'matched_keywords': ['detect', 'object'],
#             'capabilities': ['Object detection in videos', ...],
#             'priority': 9
#         }
#     ]
# }
```

## Architecture

```
User Description
      ↓
IntentClassifier (keyword matching + scoring)
      ↓
AgentCapabilityRegistry (finds matching agents)
      ↓
MultiAgentCoordinator (routes to best agent)
      ↓
Selected Agent (processes the task)
```

## Benefits

1. **Natural Language Interface** - Users describe tasks naturally
2. **Self-Documenting** - Capabilities defined at top of each agent file
3. **Easy to Extend** - Just add keywords for new capabilities
4. **Backward Compatible** - Old task_type approach still works
5. **Flexible** - Can handle ambiguous or multi-agent scenarios
6. **Discoverable** - Easy to see what each agent can do

## Future Enhancements

- LLM-based classification for complex queries
- Multi-agent workflows (task requires multiple agents)
- Confidence-based agent selection
- User feedback to improve routing
