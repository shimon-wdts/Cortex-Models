from dishka import Provider, Scope, provide

from app.clients.pas import PasClient
from app.core.config import settings


class AppProvider(Provider):
    @provide(scope=Scope.APP)
    def pas_client(self) -> PasClient:
        return PasClient(
            base_url=settings.pas_base_url,
            partner_id=settings.pas_partner_id,
            api_key=settings.pas_api_key,
        )
