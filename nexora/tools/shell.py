from .basetool import BaseTool
import subprocess
import shlex
import time

class Shell(BaseTool):
    name = "shell"

    def run(self, command, timeout=60):
        """openai.function: Executes a command in the shell. Try not to run commands that will hang or block indefinitely. If a command does hang, timeout will protect from an indefinite hang and kill it after the specified timeout.

        command,timeout

        :param str command: The command to execute.
        :param int timeout: Time in seconds to wait for command to execute, default is 60. Set to -1 to wait until completion.
        """
        if '&&' in command:
            raise Exception('Cannot use `&&` in shell call')

        self.safe(f"Run `{command}`")

        try:
            # Use shlex to properly handle splitting the command
            args = shlex.split(command)

            # Start the process
            process = subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

            # Use communicate to handle input/output and enforce the timeout
            if timeout == -1:
                timeout = None

            stdout, stderr = process.communicate(timeout=timeout)

            result = {
                "stdout": stdout.decode(),
                "stderr": stderr.decode(),
                "exit_code": process.returncode
            }

            # Return process output and error message, if any
            return result
        except subprocess.TimeoutExpired:
            return {"error": "Command timed out after {} seconds".format(timeout)}
        except Exception as e:
            return {"error": str(e)}
