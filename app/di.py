from dishka import Provider, Scope, provide

from app.clients.pas import PasClient
from app.core import config


class AppProvider(Provider):
    @provide(scope=Scope.APP)
    def pas_client(self) -> PasClient:
        return PasClient(
            base_url=config.PAS_BASE_URL,
            partner_id=config.PAS_PARTNER_ID,
            api_key=config.PAS_API_KEY,
        )
