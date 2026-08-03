class ApplicationError(Exception):
    """Base application error."""


class ConfigurationError(ApplicationError):
    """Required runtime configuration is missing or invalid."""


class DocumentNotFoundError(ApplicationError):
    """Requested document does not exist."""


class UnsupportedDocumentError(ApplicationError):
    """Uploaded document format is unsupported."""


class UploadTooLargeError(ApplicationError):
    """Uploaded document exceeds the configured size limit."""


class EngineNotReadyError(ApplicationError):
    """LightRAG engine cannot serve the requested operation."""
