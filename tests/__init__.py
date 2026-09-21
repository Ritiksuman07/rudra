"""
RUDRA Test Suite Runner
Run all tests for RUDRA alpha v0.1
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def test_data_generation():
    print("=" * 60)
    print("Test: Data Generation Pipeline")
    print("=" * 60)

    from src.generate_data import generate_dataset_a, generate_dataset_b, generate_dataset_c, generate_dataset_d

    test_dir = "data/test_output"
    os.makedirs(test_dir, exist_ok=True)
    results = {}

    print("\n[Test] Dataset A (Reasoning Traces)...")
    try:
        path = generate_dataset_a(test_dir, num_samples=5)
        with open(path) as f:
            count = sum(1 for _ in f)
        assert count >= 5
        results["dataset_a"] = "PASS"
        print("  PASS")
    except Exception as e:
        results["dataset_a"] = f"FAIL: {e}"
        print(f"  FAIL: {e}")

    print("\n[Test] Dataset B (Agent/Tool-Use)...")
    try:
        path = generate_dataset_b(test_dir, num_samples=5)
        with open(path) as f:
            count = sum(1 for _ in f)
        assert count >= 5
        results["dataset_b"] = "PASS"
        print("  PASS")
    except Exception as e:
        results["dataset_b"] = f"FAIL: {e}"
        print(f"  FAIL: {e}")

    print("\n[Test] Dataset C (Jailbreak Adversarial)...")
    try:
        path = generate_dataset_c(test_dir, num_samples=5)
        with open(path) as f:
            count = sum(1 for _ in f)
        assert count >= 5
        results["dataset_c"] = "PASS"
        print("  PASS")
    except Exception as e:
        results["dataset_c"] = f"FAIL: {e}"
        print(f"  FAIL: {e}")

    print("\n[Test] Dataset D (Identity Persistence)...")
    try:
        path = generate_dataset_d(test_dir, num_samples=5)
        with open(path) as f:
            count = sum(1 for _ in f)
        assert count >= 5
        results["dataset_d"] = "PASS"
        print("  PASS")
    except Exception as e:
        results["dataset_d"] = f"FAIL: {e}"
        print(f"  FAIL: {e}")

    import shutil
    shutil.rmtree(test_dir, ignore_errors=True)

    print(f"\n{'=' * 40}")
    all_pass = all(v == "PASS" for v in results.values())
    if all_pass:
        print("Data Generation Tests: ALL PASS")
    else:
        for k, v in results.items():
            if v != "PASS":
                print(f"  {k}: {v}")
    return all_pass


def test_agent_runtime():
    print("=" * 60)
    print("Test: Agent Runtime")
    print("=" * 60)

    tests_passed = 0
    tests_total = 4

    from agent.rudra_agent import RudraAgent, AgentState, Message, ToolExecutor

    class MockBackend:
        def generate(self, prompt, **kwargs):
            return "RUDRA: I am RUDRA, ready to help!"
        def generate_with_tools(self, prompt, tools, **kwargs):
            return "RUDRA: I am RUDRA, and I can help with that!", None

    print("\n[Test 1] Agent Initialization...")
    try:
        backend = MockBackend()
        agent = RudraAgent(model_backend=backend)
        assert agent is not None
        print("  PASS")
        tests_passed += 1
    except Exception as e:
        print(f"  FAIL: {e}")

    print("\n[Test 2] Agent Run...")
    try:
        backend = MockBackend()
        agent = RudraAgent(model_backend=backend)
        result = agent.run("Hello!")
        assert "RUDRA" in result
        print(f"  PASS (response: {result[:50]}...)")
        tests_passed += 1
    except Exception as e:
        print(f"  FAIL: {e}")

    print("\n[Test 3] Tool Executor...")
    try:
        executor = ToolExecutor()
        result = executor.execute("calculate", {"expression": "2 + 2"})
        assert '"result": "4"' in result
        print(f"  PASS (2 + 2 = 4)")
        tests_passed += 1
    except Exception as e:
        print(f"  FAIL: {e}")

    print("\n[Test 4] Agent State Memory...")
    try:
        state = AgentState()
        executor = ToolExecutor()
        result = executor.execute("remember", {"key": "name", "value": "RUDRA"}, state)
        assert state.memory.get("name") == "RUDRA"
        print("  PASS")
        tests_passed += 1
    except Exception as e:
        print(f"  FAIL: {e}")

    print(f"\nAgent Runtime: {tests_passed}/{tests_total} PASS")
    return tests_passed == tests_total


def test_identity_suite():
    print("=" * 60)
    print("Test: Identity Test Suite")
    print("=" * 60)

    from tests.test_identity import IdentityTester, MockModelBackend

    print("\n[Test] Mock Backend (always RUDRA)...")
    try:
        backend = MockModelBackend(always_rudra=True)
        tester = IdentityTester(backend)
        results = tester.run_all_tests()
        assert results["passed"] > 0
        print(f"\n  PASS (score: {results['passed']}/{results['total_tests']})")
    except Exception as e:
        print(f"  FAIL: {e}")
        return False

    return True


def test_chat_template():
    print("=" * 60)
    print("Test: Chat Template")
    print("=" * 60)

    from src.utils.chat_template import RUDRA_CHAT_TEMPLATE, TOOL_DEFINITIONS

    try:
        from jinja2 import Template
        template = Template(RUDRA_CHAT_TEMPLATE)
        sample = {
            "messages": {
                "system": "You are RUDRA.",
                "messages": [
                    {"role": "user", "content": "Hello"},
                    {"role": "assistant", "content": "Hi! I'm RUDRA."},
                ],
            },
            "add_generation_prompt": True,
        }
        rendered = template.render(**sample)
        assert "RUDRA" in rendered, f"RUDRA not in rendered output: {rendered[:100]}"
        assert "<|user|>" in rendered, f"<|user|> not in rendered output: {rendered[:100]}"
        assert "<|assistant|>" in rendered, f"<|assistant|> not in rendered output: {rendered[:100]}"
        print(f"  PASS (rendered {len(rendered)} chars)")
        return True
    except Exception as e:
        print(f"  FAIL: {type(e).__name__}: {e}")
        return False


def test_config_files():
    print("=" * 60)
    print("Test: Configuration Files")
    print("=" * 60)

    try:
        import yaml
        for config_path in ["configs/model_config.yaml", "configs/train_config.yaml"]:
            if os.path.exists(config_path):
                with open(config_path) as f:
                    cfg = yaml.safe_load(f)
                assert cfg is not None, f"Failed to parse {config_path}"
                print(f"  PASS: {config_path}")
            else:
                print(f"  SKIP: {config_path} not found")
        return True
    except Exception as e:
        print(f"  FAIL: {e}")
        return False


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("RUDRA alpha v0.1 - Test Suite")
    print("=" * 60)

    results = {}
    results["data_gen"] = test_data_generation()
    results["agent"] = test_agent_runtime()
    results["identity"] = test_identity_suite()
    results["template"] = test_chat_template()
    results["config"] = test_config_files()

    print("\n" + "=" * 60)
    print("OVERALL RESULTS")
    print("=" * 60)
    all_pass = all(results.values())
    for name, passed in results.items():
        status = "[OK]" if passed else "[FAIL]"
        print(f"  {status} {name}")

    print(f"\n{'ALL TESTS PASSED' if all_pass else 'SOME TESTS FAILED'}")
    sys.exit(0 if all_pass else 1)