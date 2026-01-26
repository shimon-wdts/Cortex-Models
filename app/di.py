from dynaconf import Dynaconf
from dishka import Provider, Scope, provide

from app.clients.pas import PasClient
from app.core.config import settings


class AppProvider(Provider):
    @provide(scope=Scope.APP)
    def dynaconf_settings(self) -> Dynaconf:
        return settings

    @provide(scope=Scope.APP)
    def pas_client(self, dynaconf_settings: Dynaconf) -> PasClient:
        return PasClient(
            base_url=dynaconf_settings.pas.base_url,
            partner_id=dynaconf_settings.pas.partner_id,
            api_key=dynaconf_settings.pas.api_key,
        )
