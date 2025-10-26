"""
Test script for intent-based routing

Run this to verify that task descriptions are correctly routed to agents.
"""

from routing.intent_classifier import get_intent_classifier
from models.agent_capabilities import AgentCapabilityRegistry
from agents.vision_agent import VisionAgent
from agents.transcription_agent import TranscriptionAgent


def test_intent_classification():
    """Test intent classification with various task descriptions"""

    # Initialize agents (this registers their capabilities)
    vision_agent = VisionAgent()
    transcription_agent = TranscriptionAgent()

    # Get classifier
    classifier = get_intent_classifier()

    # Test cases: (description, expected_agent)
    test_cases = [
        # Transcription tasks
        ("Generate the transcript for the video", "transcription_agent"),
        ("Transcribe this video", "transcription_agent"),
        ("What was said in the video?", "transcription_agent"),
        ("Create subtitles for the video", "transcription_agent"),
        ("Extract audio and convert to text", "transcription_agent"),

        # Vision/Object detection tasks
        ("Detect objects in the video", "vision_agent"),
        ("Find all people in the video", "vision_agent"),
        ("What objects appear in this video?", "vision_agent"),
        ("Identify all animals in the video", "vision_agent"),
        ("Track movement of cars", "vision_agent"),
        ("What's happening in the video?", "vision_agent"),
    ]

    print("=" * 80)
    print("INTENT CLASSIFICATION TEST")
    print("=" * 80)

    passed = 0
    failed = 0

    for description, expected_agent in test_cases:
        # Classify
        best_agent = classifier.get_best_agent(description)
        matches = classifier.classify(description)

        # Check result
        success = best_agent == expected_agent
        if success:
            passed += 1
            status = "✅ PASS"
        else:
            failed += 1
            status = "❌ FAIL"

        # Show details
        print(f"\n{status}")
        print(f"Description: '{description}'")
        print(f"Expected:    {expected_agent}")
        print(f"Got:         {best_agent}")
        if matches:
            print(f"All matches: {matches}")

    # Summary
    print("\n" + "=" * 80)
    print(f"RESULTS: {passed} passed, {failed} failed out of {len(test_cases)} tests")
    print("=" * 80)

    return failed == 0


def test_explain_routing():
    """Test the routing explanation feature"""

    # Initialize agents
    vision_agent = VisionAgent()
    transcription_agent = TranscriptionAgent()

    classifier = get_intent_classifier()

    print("\n" + "=" * 80)
    print("ROUTING EXPLANATION TEST")
    print("=" * 80)

    test_descriptions = [
        "Transcribe the video and detect all people",
        "What was said in the video?",
        "Find all objects in the video",
    ]

    for description in test_descriptions:
        print(f"\n📝 Description: '{description}'")
        explanation = classifier.explain_routing(description)

        print(f"Matches found: {len(explanation['matches'])}")
        for match in explanation['matches']:
            print(f"  • {match['agent']}: score={match['score']:.2f}")
            print(f"    Matched keywords: {match['matched_keywords']}")
            print(f"    Priority: {match['priority']}")


def test_capability_registry():
    """Test the capability registry"""

    # Initialize agents
    vision_agent = VisionAgent()
    transcription_agent = TranscriptionAgent()

    print("\n" + "=" * 80)
    print("CAPABILITY REGISTRY TEST")
    print("=" * 80)

    all_capabilities = AgentCapabilityRegistry.get_all_capabilities()

    print(f"\nRegistered agents: {len(all_capabilities)}")
    for agent_name, capability in all_capabilities.items():
        print(f"\n🤖 {agent_name}:")
        print(f"  Capabilities: {capability.capabilities}")
        print(f"  Categories: {capability.categories}")
        print(f"  Priority: {capability.routing_priority}")
        print(f"  Example keywords: {capability.intent_keywords[:5]}...")


if __name__ == "__main__":
    print("\n🧪 Starting Intent Routing Tests...\n")

    # Run tests
    test_capability_registry()
    test_explain_routing()
    success = test_intent_classification()

    if success:
        print("\n✅ All tests passed!")
    else:
        print("\n❌ Some tests failed!")
