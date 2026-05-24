# Custom exception for PerceptionMetrics.

class PerceptionMetricsException(Exception):
    """Wraps any exception with the file name and line number it occurred on.

    :param error_message: The original exception caught in the except block.
    :type error_message: Exception
    """

    def __init__(self, error_message: Exception) -> None:
        super().__init__(str(error_message))
        self.error_message = error_message

        # e.__traceback__ works whether the exception was raised manually
        # (raise FileNotFoundError) or caught from a library call.
        # It does not depend on sys.exc_info() being active, so it always
        # returns the correct file and line even when called from a helper.
        tb = getattr(error_message, "__traceback__", None)

        if tb is not None:
            self.lineno    = tb.tb_lineno
            self.file_name = tb.tb_frame.f_code.co_filename
        else:
            self.lineno    = -1
            self.file_name = "<unknown>"

    def __str__(self) -> str:
        return (
            f"Error in [{self.file_name}] "
            f"at line [{self.lineno}]: "
            f"{self.error_message}"
        )