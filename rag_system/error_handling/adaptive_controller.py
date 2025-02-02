from typing import Any, Dict
from .base_controller import BaseErrorController

class AdaptiveErrorController(BaseErrorController):
    def __init__(self):
        super().__init__()
        self.error_count = {}

    def handle_error(self, error: Exception, context: Dict[str, Any]) -> Dict[str, Any]:
        error_type = type(error).__name__
        self.error_count[error_type] = self.error_count.get(error_type, 0) + 1

        if self.error_count[error_type] > 3:
            # If we've seen this error more than 3 times, try a different approach
            return self.advanced_error_handling(error, context)
        else:
            # Otherwise, use the basic error handling
            return self.basic_error_handling(error, context)

    def basic_error_handling(self, error: Exception, context: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "error_type": type(error).__name__,
            "error_message": str(error),
            "context": context,
            "action": "retry",
        }

    def advanced_error_handling(self, error: Exception, context: Dict[str, Any]) -> Dict[str, Any]:
        # Implement more sophisticated error handling here
        # This could involve changing the approach, using a different model, etc.
        return {
            "error_type": type(error).__name__,
            "error_message": str(error),
            "context": context,
            "action": "change_approach",
            "new_approach": "Use a different model or technique",
        }

    def reset(self):
        self.error_count.clear()
