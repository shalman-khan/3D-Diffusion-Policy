from typing import Dict
from diffusion_policy_3d.env_runner.base_runner import BaseRunner
from diffusion_policy_3d.policy.base_policy import BasePolicy


class RealRobotRunner(BaseRunner):
    """
    No-op runner for real robot training.
    Sim rollouts are skipped — evaluation is done via policy_executor.py on the real robot.
    Returns an empty dict so training continues without error.
    """

    def __init__(self, output_dir, **kwargs):
        super().__init__(output_dir)

    def run(self, policy: BasePolicy) -> Dict:
        # test_mean_score required by checkpoint manager — no sim rollouts for real robot
        return {"test_mean_score": 0.0}
