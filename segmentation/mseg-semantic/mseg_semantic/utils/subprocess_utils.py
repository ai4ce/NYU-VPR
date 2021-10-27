#!/usr/bin/python3

import subprocess
from typing import Optional, Tuple

def run_command(cmd: str, return_output: bool = False) -> Optional[Tuple[bytes, bytes]]:
    """
    Block until system call completes

    Args:
        cmd: string, representing shell command

    Returns:
        Tuple of (stdout, stderr) output if return_output is True, else None
    """
    (stdout_data, stderr_data) = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE).communicate()

    if return_output:
        return stdout_data, stderr_data
    return None