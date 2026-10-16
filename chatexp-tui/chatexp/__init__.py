"""ChatExp package."""

__all__ = ["ChatExpService"]


def __getattr__(name: str):
    if name == "ChatExpService":
        from .service import ChatExpService

        return ChatExpService
    raise AttributeError(name)
