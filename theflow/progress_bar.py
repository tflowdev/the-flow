import uuid
from typing import Dict

import tqdm

try:
    from ray.air import session
except ImportError:
    session = None


class The FlowProgressBarActions:
    CREATE = "create"
    UPDATE = "update"
    SET_POSTFIX = "set_postfix"
    CLOSE = "close"


class The FlowProgressBar:
    """Class for progress bars that supports distributed progress bars in ray.

    # Inputs

    :param report_to_ray: (bool) use the ray.air.session method
        to report progress to the ray driver. If false then this behaves as a normal tqdm
        progress bar
    :param config: (dict) the tqdm configs used for the progress bar. See https://github.com/tqdm/tqdm#parameters
        for list of parameters
    :param is_coordinator: (bool) whether the calling process is the coordinator process.

    # Example usage:

    ```python
    from theflow.progress_bar import The FlowProgressBar

    config = {"total": 20, "desc": "Sample progress bar"}
    pbar = The FlowProgressBar(report_to_ray=False, config=config, is_coordinator=True)
    for i in range(20):
        pbar.update(1)
    pbar.close()
    ```
    """

    def __init__(
        self,
        report_to_ray: bool,
        config: Dict,
        is_coordinator: bool,
    ) -> None:
        """Constructor for the The FlowProgressBar class.

        # Inputs

        :param report_to_ray: (bool) use the ray.air.session method
            to report progress to the ray driver. If false then this behaves as a normal tqdm
            progress bar
        :param config: (dict) the tqdm configs used for the progress bar. See https://github.com/tqdm/tqdm#parameters
            for list of parameters
        :param is_coordinator: (bool) whether the calling process is the coordinator process.

        # Return

        :return: (None) `None`
        """
        if report_to_ray and session is None:
            raise ValueError("Set report_to_ray=True but ray is not installed. Run `pip install ray`")

        self.id = str(uuid.uuid4())[-8:]

        self.report_to_ray = report_to_ray
        self.is_coordinator = is_coordinator
        self.config = config

        self.total_steps = 0
        self.progress_bar = None

        if not self.report_to_ray:
            if self.is_coordinator:
                self.progress_bar = tqdm.tqdm(**config)
        else:
            if "file" in self.config:
                self.config.pop("file")
            # All processes need to call ray.train.report since ray has a lock that blocks
            # a process when calling report if there are processes that haven't called it. Similar
            # to a distributed checkpoint. Therefore we pass the flag to the driver
            session.report(
                metrics={
                    "progress_bar": {
                        "id": self.id,
                        "config": self.config,
                        "action": The FlowProgressBarActions.CREATE,
                        "is_coordinator": self.is_coordinator,
                    }
                }
            )

    def set_postfix(self, postfix: Dict):
        if self.progress_bar:
            self.progress_bar.set_postfix(postfix)
        elif self.report_to_ray:
            session.report(
                metrics={
                    "progress_bar": {
                        "id": self.id,
                        "postfix": postfix,
                        "is_coordinator": self.is_coordinator,
                        "action": The FlowProgressBarActions.SET_POSTFIX,
                    }
                }
            )

    def update(self, steps: int) -> None:
        """Updates the progress bar.

        # Inputs

        :param steps: (int) number of steps to update the progress bar by

        # Return

        :return: (None) `None`
        """
        self.total_steps += steps
        if self.progress_bar:
            self.progress_bar.update(steps)
        elif self.report_to_ray:
            session.report(
                metrics={
                    "progress_bar": {
                        "id": self.id,
                        "update_by": steps,
                        "is_coordinator": self.is_coordinator,
                        "action": The FlowProgressBarActions.UPDATE,
                    }
                }
            )

    def close(self) -> None:
        """Closes the progress bar.

        # Return

        :return: (None) `None`
        """
        if self.progress_bar:
            self.progress_bar.close()
        elif self.report_to_ray:
            session.report(
                metrics={
                    "progress_bar": {
                        "id": self.id,
                        "is_coordinator": self.is_coordinator,
                        "action": The FlowProgressBarActions.CLOSE,
                    }
                }
            )
