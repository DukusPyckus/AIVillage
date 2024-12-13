from asyncio.log import logger
from typing import List, Dict, Any
from agents.unified_base_agent import UnifiedBaseAgent
from communications.protocol import StandardCommunicationProtocol, Message, MessageType
from core.config import UnifiedConfig
from ..magi.magi_agent import MagiAgent
from ..sage.sage_agent import SageAgent
from rag_system.error_handling.error_handler import error_handler, AIVillageException
from .analytics.unified_analytics import UnifiedAnalytics

class KingCoordinator:
    def __init__(self, config: UnifiedConfig, communication_protocol: StandardCommunicationProtocol):
        self.config = config
        self.communication_protocol = communication_protocol
        self.agents: Dict[str, UnifiedBaseAgent] = {}
        self.task_manager = self.router = self.decision_maker = self.problem_analyzer = self.king_agent = None
        self.unified_analytics = UnifiedAnalytics()

    def add_agent(self, agent_name: str, agent: UnifiedBaseAgent):
        self.agents[agent_name] = agent

    @error_handler.handle_error
    async def coordinate_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
        start_time = self.unified_analytics.get_current_time()
        result = await self._delegate_task(task)
        execution_time = self.unified_analytics.get_current_time() - start_time

        self.unified_analytics.record_task_completion(task['id'], execution_time, result.get('success', False))
        self.unified_analytics.record_metric(f"task_type_{task['type']}_execution_time", execution_time)

        return result

    @error_handler.handle_error
    async def _delegate_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
        agent = self._get_agent_for_task(task['type'])
        if agent:
            return await agent.execute_task(task)

        raise ValueError("No suitable agent found for the task")

    def _get_agent_for_task(self, task_type: str) -> UnifiedBaseAgent:
        if task_type == 'research':
            return next((agent for agent in self.agents.values() if isinstance(agent, SageAgent)), None)
        if task_type in ['coding', 'debugging', 'code_review']:
            return next((agent for agent in self.agents.values() if isinstance(agent, MagiAgent)), None)
        return next(iter(self.agents.values()), None)

    async def handle_message(self, message: Message):
        if message.type == MessageType.TASK:
            result = await self.coordinate_task(message.content)
            await self._send_response(message, result)
            await self.task_manager.assign_task(message.content)

    async def _send_response(self, message: Message, result: Dict[str, Any]):
        response = Message(
            type=MessageType.RESPONSE,
            sender="KingCoordinator",
            receiver=message.sender,
            content=result,
            parent_id=message.id
        )
        await self.communication_protocol.send_message(response)

    async def process_task_completion(self, task: Dict[str, Any], result: Any):
        try:
            updates = [
                self.router.train_model([{'task': task['description'], 'assigned_agent': task['assigned_agents'][0]}]),
                self.task_manager.complete_task(task['id'], result),
                self.decision_maker.update_model(task, result),
                self.problem_analyzer.update_models(task, result),
                self.decision_maker.update_mcts(task, result),
                self.king_agent.update(task, result)
            ]
            await asyncio.gather(*updates)

            self._record_task_metrics(task, result)
        except Exception as e:
            logger.error(f"Error processing task completion: {str(e)}")
            raise AIVillageException(f"Error processing task completion: {str(e)}")

    def _record_task_metrics(self, task: Dict[str, Any], result: Any):
        self.unified_analytics.record_metric(f"task_type_{task['type']}_success", int(result.get('success', False)))
        self.unified_analytics.record_metric(f"agent_{task['assigned_agents'][0]}_performance", result.get('performance', 0.5))

    async def save_models(self, path: str):
        await self._perform_model_action("save", path)

    async def load_models(self, path: str):
        await self._perform_model_action("load", path)

    async def _perform_model_action(self, action: str, path: str):
        try:
            getattr(self.router, action)(f"{path}/agent_router.pt")
            actions = [
                getattr(self.decision_maker, f"{action}_models")(f"{path}/decision_maker"),
                getattr(self.task_manager, f"{action}_models")(f"{path}/task_manager"),
                getattr(self.problem_analyzer, f"{action}_models")(f"{path}/problem_analyzer")
            ]
            await asyncio.gather(*actions)
            logger.info(f"Models {action}d from {path}")
        except Exception as e:
            logger.error(f"Error {action}ing models: {str(e)}")
            raise AIVillageException(f"Error {action}ing models: {str(e)}")

    async def introspect(self) -> Dict[str, Any]:
        return {
            "agents": list(self.agents.keys()),
            "router_info": self.router.introspect(),
            "decision_maker_info": await self.decision_maker.introspect(),
            "task_manager_info": await self.task_manager.introspect(),
            "problem_analyzer_info": await self.problem_analyzer.introspect(),
            "analytics_summary": self.unified_analytics.generate_summary_report()
        }
