from .base import BaseSpecBuilder
from ..constants import EIP6110


class GasperSiestaSpecBuilder(BaseSpecBuilder):
    fork: str = EIP6110

    @classmethod
    def imports(cls, preset_name: str):
        return f'''
from eth2spec.phase0 import {preset_name} as phase0
'''
