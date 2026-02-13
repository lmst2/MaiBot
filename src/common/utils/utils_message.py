import msgpack

from src.common.data_models.message_component_model import MessageSequence


class MessageUtils:
    @staticmethod
    def from_db_record_msg_to_MaiSeq(raw_content: bytes) -> MessageSequence:
        unpacked_data = msgpack.unpackb(raw_content)
        return MessageSequence.from_dict(unpacked_data)

    @staticmethod
    async def from_MaiSeq_to_db_record_msg(msg: MessageSequence) -> bytes:
        dict_representation = msg.to_dict()
        return msgpack.packb(dict_representation)  # type: ignore
