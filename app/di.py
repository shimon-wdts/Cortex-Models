from dishka import Provider, Scope, provide

from app.clients.pas import PasClient
from app.core import config


class AppProvider(Provider):
    @provide(scope=Scope.APP)
    def pas_client(self) -> PasClient:
        return PasClient(
            base_url=config.settings.pas_base_url,
            partner_id=config.settings.pas_partner_id,
            api_key=config.settings.pas_api_key,
        )
