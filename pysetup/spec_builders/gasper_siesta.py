from .base import BaseSpecBuilder
from ..constants import GASPER_SIESTA


class GasperSiestaSpecBuilder(BaseSpecBuilder):
    fork: str = GASPER_SIESTA

    @classmethod
    def imports(cls, preset_name: str):
        return super().imports(preset_name) + f'''
from eth2spec.phase0 import {preset_name} as phase0
'''
