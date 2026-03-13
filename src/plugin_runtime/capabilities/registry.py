from typing import TYPE_CHECKING

from src.common.logger import get_logger
from src.plugin_runtime.host.supervisor import PluginSupervisor

if TYPE_CHECKING:
    from src.plugin_runtime.integration import PluginRuntimeManager

logger = get_logger("plugin_runtime.integration")


def register_capability_impls(manager: "PluginRuntimeManager", supervisor: PluginSupervisor) -> None:
    """向指定 Supervisor 注册主程序提供的能力实现。"""
    cap_service = supervisor.capability_service

    cap_service.register_capability("send.text", manager._cap_send_text)
    cap_service.register_capability("send.emoji", manager._cap_send_emoji)
    cap_service.register_capability("send.image", manager._cap_send_image)
    cap_service.register_capability("send.command", manager._cap_send_command)
    cap_service.register_capability("send.custom", manager._cap_send_custom)

    cap_service.register_capability("llm.generate", manager._cap_llm_generate)
    cap_service.register_capability("llm.generate_with_tools", manager._cap_llm_generate_with_tools)
    cap_service.register_capability("llm.get_available_models", manager._cap_llm_get_available_models)

    cap_service.register_capability("config.get", manager._cap_config_get)
    cap_service.register_capability("config.get_plugin", manager._cap_config_get_plugin)
    cap_service.register_capability("config.get_all", manager._cap_config_get_all)

    cap_service.register_capability("database.query", manager._cap_database_query)
    cap_service.register_capability("database.save", manager._cap_database_save)
    cap_service.register_capability("database.get", manager._cap_database_get)
    cap_service.register_capability("database.delete", manager._cap_database_delete)
    cap_service.register_capability("database.count", manager._cap_database_count)

    cap_service.register_capability("chat.get_all_streams", manager._cap_chat_get_all_streams)
    cap_service.register_capability("chat.get_group_streams", manager._cap_chat_get_group_streams)
    cap_service.register_capability("chat.get_private_streams", manager._cap_chat_get_private_streams)
    cap_service.register_capability("chat.get_stream_by_group_id", manager._cap_chat_get_stream_by_group_id)
    cap_service.register_capability("chat.get_stream_by_user_id", manager._cap_chat_get_stream_by_user_id)

    cap_service.register_capability("message.get_by_time", manager._cap_message_get_by_time)
    cap_service.register_capability("message.get_by_time_in_chat", manager._cap_message_get_by_time_in_chat)
    cap_service.register_capability("message.get_recent", manager._cap_message_get_recent)
    cap_service.register_capability("message.count_new", manager._cap_message_count_new)
    cap_service.register_capability("message.build_readable", manager._cap_message_build_readable)

    cap_service.register_capability("person.get_id", manager._cap_person_get_id)
    cap_service.register_capability("person.get_value", manager._cap_person_get_value)
    cap_service.register_capability("person.get_id_by_name", manager._cap_person_get_id_by_name)

    cap_service.register_capability("emoji.get_by_description", manager._cap_emoji_get_by_description)
    cap_service.register_capability("emoji.get_random", manager._cap_emoji_get_random)
    cap_service.register_capability("emoji.get_count", manager._cap_emoji_get_count)
    cap_service.register_capability("emoji.get_emotions", manager._cap_emoji_get_emotions)
    cap_service.register_capability("emoji.get_all", manager._cap_emoji_get_all)
    cap_service.register_capability("emoji.get_info", manager._cap_emoji_get_info)
    cap_service.register_capability("emoji.register", manager._cap_emoji_register)
    cap_service.register_capability("emoji.delete", manager._cap_emoji_delete)

    cap_service.register_capability("frequency.get_current_talk_value", manager._cap_frequency_get_current_talk_value)
    cap_service.register_capability("frequency.set_adjust", manager._cap_frequency_set_adjust)
    cap_service.register_capability("frequency.get_adjust", manager._cap_frequency_get_adjust)

    cap_service.register_capability("tool.get_definitions", manager._cap_tool_get_definitions)

    cap_service.register_capability("component.get_all_plugins", manager._cap_component_get_all_plugins)
    cap_service.register_capability("component.get_plugin_info", manager._cap_component_get_plugin_info)
    cap_service.register_capability("component.list_loaded_plugins", manager._cap_component_list_loaded_plugins)
    cap_service.register_capability("component.list_registered_plugins", manager._cap_component_list_registered_plugins)
    cap_service.register_capability("component.enable", manager._cap_component_enable)
    cap_service.register_capability("component.disable", manager._cap_component_disable)
    cap_service.register_capability("component.load_plugin", manager._cap_component_load_plugin)
    cap_service.register_capability("component.unload_plugin", manager._cap_component_unload_plugin)
    cap_service.register_capability("component.reload_plugin", manager._cap_component_reload_plugin)

    cap_service.register_capability("knowledge.search", manager._cap_knowledge_search)
    logger.debug("已注册全部主程序能力实现")
