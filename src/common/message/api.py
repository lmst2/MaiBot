from src.common.server import get_global_server
import os
import importlib.metadata
from maim_message import MessageServer
from src.common.logger import get_logger
from src.config.config import global_config

global_api = None


def get_global_api() -> MessageServer:  # sourcery skip: extract-method
    """获取全局MessageServer实例"""
    global global_api
    if global_api is None:
        # 检查maim_message版本
        try:
            maim_message_version = importlib.metadata.version("maim_message")
            version_int = [int(x) for x in maim_message_version.split(".")]
            version_compatible = version_int >= [0, 3, 3]
            # Check for API Server feature (>= 0.6.0)
            has_api_server_feature = version_int >= [0, 6, 2]
        except (importlib.metadata.PackageNotFoundError, ValueError):
            version_compatible = False
            has_api_server_feature = False

        # 读取配置项
        maim_message_config = global_config.maim_message

        # 设置基本参数 (Legacy Server Mode)
        kwargs = {
            "host": os.environ["HOST"],
            "port": int(os.environ["PORT"]),
            "app": get_global_server().get_app(),
        }

        # 只有在版本 >= 0.3.0 时才使用高级特性
        if version_compatible:
            # 添加自定义logger
            maim_message_logger = get_logger("maim_message")
            kwargs["custom_logger"] = maim_message_logger

            # 添加token认证
            if maim_message_config.auth_token and len(maim_message_config.auth_token) > 0:
                kwargs["enable_token"] = True

            # Removed legacy custom config block (use_custom) as requested.
            kwargs["enable_custom_uvicorn_logger"] = False

        global_api = MessageServer(**kwargs)
        if version_compatible and maim_message_config.auth_token:
            for token in maim_message_config.auth_token:
                global_api.add_valid_token(token)

        # ---------------------------------------------------------------------
        # Additional API Server Configuration (maim_message >= 6.0)
        # ---------------------------------------------------------------------
        enable_api_server = maim_message_config.enable_api_server
        
        # 如果版本支持且启用了API Server，则初始化额外服务器
        if has_api_server_feature and enable_api_server:
            try:
                from maim_message.server import WebSocketServer, ServerConfig
                from maim_message.message import APIMessageBase

                api_logger = get_logger("maim_message_api_server")

                # 1. Prepare Config
                api_server_host = maim_message_config.api_server_host
                api_server_port = maim_message_config.api_server_port
                use_wss = maim_message_config.api_server_use_wss

                server_config = ServerConfig(
                    host=api_server_host,
                    port=api_server_port,
                    ssl_enabled=use_wss,
                    ssl_certfile=maim_message_config.api_server_cert_file if use_wss else None,
                    ssl_keyfile=maim_message_config.api_server_key_file if use_wss else None,
                    custom_logger=api_logger  # 传入自定义logger
                )

                # 2. Setup Auth Handler
                async def auth_handler(metadata: dict) -> bool:
                    allowed_keys = maim_message_config.api_server_allowed_api_keys
                    # If list is empty/None, allow all (default behavior of returning True)
                    if not allowed_keys:
                        return True
                    
                    api_key = metadata.get("api_key")
                    if api_key in allowed_keys:
                        return True
                    
                    api_logger.warning(f"Rejected connection with invalid API Key: {api_key}")
                    return False

                server_config.on_auth = auth_handler

                # 3. Setup Message Bridge
                # Initialize refined route map if not exists
                if not hasattr(global_api, "platform_map"):
                    global_api.platform_map = {}

                async def bridge_message_handler(message: APIMessageBase, metadata: dict):
                    # 使用 MessageConverter 转换 APIMessageBase 到 Legacy MessageBase
                    # 接收场景：收到从 Adapter 转发的外部消息
                    # sender_info 包含消息发送者信息，需要提取到 group_info/user_info
                    from maim_message import MessageConverter
                    
                    legacy_message = MessageConverter.from_api_receive(message)
                    msg_dict = legacy_message.to_dict()
                    
                    # Compatibility Layer: Ensure format_info exists with defaults
                    # MaiMBot's check_types() accesses format_info.accept_format without None check
                    if "message_info" in msg_dict:
                        msg_info = msg_dict["message_info"]
                        
                        if "format_info" not in msg_info or msg_info["format_info"] is None:
                            msg_info["format_info"] = {
                                "content_format": ["text", "image", "emoji", "voice"],
                                "accept_format": [
                                    "text", "image", "emoji", "reply", "voice", "command",
                                    "voiceurl", "music", "videourl", "file", "imageurl", "forward", "video"
                                ]
                            }
                                
                        # Route Caching Logic: Map platform to API Key (or connection uuid as fallback)
                        # This allows us to send messages back to the correct API client for this platform
                        try:
                            # Get api_key from metadata, use uuid as fallback if api_key is empty
                            api_key = metadata.get("api_key") or metadata.get("uuid") or "unknown"
                            platform = msg_info.get("platform")
                            api_logger.debug(f"Bridge received: api_key='{api_key}', platform='{platform}'")
                            
                            if platform:
                                global_api.platform_map[platform] = api_key
                                api_logger.info(f"Updated platform_map: {platform} -> {api_key}")
                        except Exception as e:
                            api_logger.warning(f"Failed to update platform map: {e}")

                    # Compatibility Layer: Ensure raw_message exists (even if None) as it's part of MessageBase
                    if "raw_message" not in msg_dict:
                        msg_dict["raw_message"] = None
                    
                    await global_api.process_message(msg_dict)

                server_config.on_message = bridge_message_handler

                # 3.5. Register custom message handlers (bridge to Legacy handlers)
                # message_id_echo: handles message ID echo from adapters
                # 兼容新旧两个版本的 maim_message:
                # - 旧版: handler(payload)
                # - 新版: handler(payload, metadata)
                async def custom_message_id_echo_handler(payload: dict, metadata: dict = None):
                    # Bridge to the Legacy custom handler registered in main.py
                    try:
                        # The Legacy handler expects the payload format directly
                        if hasattr(global_api, '_custom_message_handlers'):
                            handler = global_api._custom_message_handlers.get("message_id_echo")
                            if handler:
                                await handler(payload)
                                api_logger.debug(f"Processed message_id_echo: {payload}")
                            else:
                                api_logger.debug(f"No handler for message_id_echo, payload: {payload}")
                    except Exception as e:
                        api_logger.warning(f"Failed to process message_id_echo: {e}")

                server_config.register_custom_handler("message_id_echo", custom_message_id_echo_handler)

                # 4. Initialize Server
                extra_server = WebSocketServer(config=server_config)
                
                # 5. Patch global_api lifecycle methods to manage both servers
                original_run = global_api.run
                original_stop = global_api.stop

                async def patched_run():
                    api_logger.info(f"Starting Additional API Server on {api_server_host}:{api_server_port} (WSS: {use_wss})")
                    # Start the extra server (non-blocking start)
                    await extra_server.start()
                    # Run the original legacy server (this usually keeps running)
                    await original_run()

                async def patched_stop():
                    api_logger.info("Stopping Additional API Server...")
                    await extra_server.stop()
                    await original_stop()

                global_api.run = patched_run
                global_api.stop = patched_stop
                
                # Attach for reference
                global_api.extra_server = extra_server

            except ImportError:
                get_logger("maim_message").error("Cannot import maim_message.server components. Is maim_message >= 0.6.0 installed?")
            except Exception as e:
                get_logger("maim_message").error(f"Failed to initialize Additional API Server: {e}")
                import traceback
                get_logger("maim_message").debug(traceback.format_exc())

    return global_api

