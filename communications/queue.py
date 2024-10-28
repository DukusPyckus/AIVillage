"""Enhanced message queue with priority handling and statistics."""

from typing import Dict, List, Optional, Set
from collections import deque
from datetime import datetime
from .message import Message, Priority

class MessageQueue:
    """Enhanced message queue with priority handling and statistics."""
    def __init__(self):
        self._queues: Dict[Priority, deque[Message]] = {
            Priority.LOW: deque(),
            Priority.MEDIUM: deque(),
            Priority.HIGH: deque()
        }
        self.last_processed: datetime = datetime.now()
        self.stats = {
            'messages_processed': 0,
            'messages_by_priority': {
                Priority.LOW: 0,
                Priority.MEDIUM: 0,
                Priority.HIGH: 0
            }
        }

    async def enqueue(self, message: Message) -> None:
        """Add a message to the appropriate priority queue."""
        self._queues[message.priority].append(message)
        self.stats['messages_by_priority'][message.priority] += 1

    async def dequeue(self) -> Optional[Message]:
        """Get the next message based on priority."""
        for priority in reversed(list(Priority)):
            if self._queues[priority]:
                self.last_processed = datetime.now()
                self.stats['messages_processed'] += 1
                return self._queues[priority].popleft()
        return None

    async def is_empty(self) -> bool:
        """Check if there are any messages in the queue."""
        return all(len(queue) == 0 for queue in self._queues.values())

    async def get_messages_by_priority(self, priority: Priority) -> List[Message]:
        """Get all messages of a specific priority."""
        return list(self._queues[priority])

    async def get_all_messages(self) -> List[Message]:
        """Get all messages in priority order (high to low)."""
        all_messages = []
        for priority in reversed(list(Priority)):
            messages = await self.get_messages_by_priority(priority)
            all_messages.extend(messages)
        return all_messages

    async def get_queue_stats(self) -> Dict[str, int]:
        """Get statistics about the queue state."""
        return {
            'high_priority': len(self._queues[Priority.HIGH]),
            'medium_priority': len(self._queues[Priority.MEDIUM]),
            'low_priority': len(self._queues[Priority.LOW]),
            'total': sum(len(queue) for queue in self._queues.values()),
            'total_processed': self.stats['messages_processed']
        }
