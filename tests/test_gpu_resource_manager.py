import threading
import time
import unittest

from services.gpu_resource_manager import gpu_inference_slot, is_gpu_inference_busy


class GpuResourceManagerTestCase(unittest.TestCase):
    def test_slot_serializes_two_owners(self):
        events: list[str] = []
        first_started = threading.Event()
        second_finished = threading.Event()

        def first_owner() -> None:
            with gpu_inference_slot("first"):
                events.append("first-start")
                first_started.set()
                time.sleep(0.05)
                events.append("first-end")

        def second_owner() -> None:
            first_started.wait(timeout=1)
            with gpu_inference_slot("second"):
                events.append("second-start")
            second_finished.set()

        first_thread = threading.Thread(target=first_owner)
        second_thread = threading.Thread(target=second_owner)
        first_thread.start()
        second_thread.start()
        first_thread.join(timeout=1)
        second_thread.join(timeout=1)

        self.assertTrue(second_finished.is_set())
        self.assertEqual(events, ["first-start", "first-end", "second-start"])
        self.assertFalse(is_gpu_inference_busy())

    def test_slot_is_released_after_exception(self):
        with self.assertRaisesRegex(RuntimeError, "expected"):
            with gpu_inference_slot("failing"):
                self.assertTrue(is_gpu_inference_busy())
                raise RuntimeError("expected")

        self.assertFalse(is_gpu_inference_busy())


if __name__ == "__main__":
    unittest.main()
