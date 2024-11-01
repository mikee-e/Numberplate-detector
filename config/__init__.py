import json
import os


class Config(dict):
    def __init__(self, path_to_cfg: str) -> None:
        self._path = path_to_cfg
        with open(self._path, "r", encoding="utf-8") as cfg_file:
            data = json.load(cfg_file)
        super().__init__(data)

    def upload(self) -> bool:
        try:
            with open(self._path, "w", encoding="utf-8") as cfg_file:
                json.dump(obj=self, fp=cfg_file, indent=4)
            return True
        except Exception:
            return False


general_cfg = Config(os.path.join(os.path.dirname(__file__), "general.json"))
numberplate_cfg = Config(os.path.join(os.path.dirname(__file__), "numberplate.json"))
