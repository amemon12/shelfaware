from __future__ import annotations

import secrets
from contextlib import asynccontextmanager
from dataclasses import replace
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Response
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.types import ASGIApp, Receive, Scope, Send

from .fixtures import fixtures
from .models import CapturedFrame, ConfirmRequest, ContextRequest, CreateSession, ReviewRequest, Session, TurnRequest, TurnResponse
from .providers import Providers, Settings
from .service import ShoppingService
from .conversation_brain import BrainSettings
from .conversation_transport import ConversationConfig, ConversationCoordinator, create_conversation_router

VERSION = '0.2.0-conversation'
BODY_LIMIT = 6 * 1024 * 1024


class RequestBoundary:
    """Check optional local API authorization and cap body bytes before JSON parsing."""
    def __init__(self, app: ASGIApp, token: str):
        self.app = app
        self.token = token

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope['type'] != 'http':
            await self.app(scope, receive, send)
            return
        headers = {k.lower(): v for k, v in scope.get('headers', [])}
        if self.token and scope['path'] != '/api/health':
            actual = headers.get(b'authorization', b'').decode('latin-1')
            if not secrets.compare_digest(actual, f'Bearer {self.token}'):
                await JSONResponse({'detail': 'API authorization required.'}, status_code=401)(scope, receive, send)
                return
        try:
            declared = int(headers.get(b'content-length', b'0'))
        except ValueError:
            declared = BODY_LIMIT + 1
        if declared > BODY_LIMIT:
            await JSONResponse({'detail': 'Request exceeds the 6 MiB limit.'}, status_code=413)(scope, receive, send)
            return
        messages, size = [], 0
        while True:
            message = await receive()
            if message['type'] == 'http.disconnect':
                return
            size += len(message.get('body', b''))
            if size > BODY_LIMIT:
                await JSONResponse({'detail': 'Request exceeds the 6 MiB limit.'}, status_code=413)(scope, receive, send)
                return
            messages.append(message)
            if not message.get('more_body', False):
                break
        index = 0

        async def buffered_receive():
            nonlocal index
            if index < len(messages):
                item = messages[index]
                index += 1
                return item
            return await receive()

        await self.app(scope, buffered_receive, send)


def create_app(settings: Settings | None = None, providers: Providers | None = None) -> FastAPI:
    # Explicit path; never depends on the process working directory.
    load_dotenv(Path(__file__).resolve().parents[2] / '.env', override=False)
    settings = settings or Settings.from_env()
    service = ShoppingService(providers or Providers(settings))
    conversation_config = replace(ConversationConfig.from_env(), api_token=settings.api_token)
    conversation_brain_settings = BrainSettings.from_env()
    brain_factory = None
    if conversation_config.product_mode == 'product_search':
        def brain_factory(profile, recipient_id):
            from .conversation_product_search import ProductSearchBrain
            return ProductSearchBrain(service, profile, recipient_id)
    conversation = ConversationCoordinator(service, config=conversation_config, brain_factory=brain_factory)

    @asynccontextmanager
    async def lifespan(_api: FastAPI):
        try:
            yield
        finally:
            try:
                await conversation.close()
            finally:
                for record in service.records.values():
                    service.clear_frame(record)
                await service.providers.close()

    api = FastAPI(title='Shelf Aware API', version=VERSION, lifespan=lifespan,
                  description='Local prototype: verified nutrition, explicit portions, and confirmed lists.')
    api.state.service = service
    api.state.conversation = conversation
    api.add_middleware(RequestBoundary, token=settings.api_token)
    api.include_router(create_conversation_router(conversation))

    @api.exception_handler(RequestValidationError)
    async def validation_error_handler(_request, exc: RequestValidationError):
        fields = ', '.join(sorted({'.'.join(str(v) for v in error['loc'][1:]) for error in exc.errors()}))
        return JSONResponse({'detail': f'Invalid request fields: {fields}. Check values, units, and required fields.'}, status_code=422)

    @api.get('/api/health')
    async def health():
        product_mode = conversation_config.product_mode == 'product_search'
        reasoning_provider = settings.text_provider if product_mode else conversation_brain_settings.provider
        reasoning_key = ({'anthropic': settings.anthropic_key, 'deepseek': settings.deepseek_key,
                          'gemini': settings.gemini_key}.get(reasoning_provider, '')
                         if product_mode else conversation_brain_settings.reasoning_key)
        return {'status': 'ok', 'providers': {
            'perceptron': bool(settings.perceptron_key),
            'gemini': 'gemini' in (settings.text_provider, settings.search_provider) and bool(settings.gemini_key),
            'deepseek': 'deepseek' in (settings.text_provider, settings.search_provider) and bool(settings.deepseek_key),
            'anthropic': settings.text_provider == 'anthropic' and bool(settings.anthropic_key),
        }, 'text_provider': settings.text_provider, 'catalog_search_provider': settings.search_provider,
        'version': VERSION, 'conversation': {
            'websocket': '/api/conversation/ws', 'input': 'pcm_s16le_24000_mono',
            'reasoning_provider': reasoning_provider,
            'transcription_provider': conversation_config.transcription_provider,
            'speech_provider': conversation_config.speech_provider,
            'speech_model': (conversation_config.elevenlabs_tts_model
                             if conversation_config.speech_provider == 'elevenlabs'
                             and conversation_config.elevenlabs_tts_model in
                             ('eleven_flash_v2_5', 'eleven_multilingual_v2', 'eleven_v4_turbo') else None),
            'analysis_mode': conversation_config.product_mode,
            'evidence_scope': ('unverified_catalog_search' if conversation_config.product_mode == 'product_search'
                               else 'reviewed_catalog'),
            'configured_not_provider_tested': bool(reasoning_key),
            'providers_configured': {
                'deepseek': reasoning_provider == 'deepseek' and bool(reasoning_key),
                'anthropic': reasoning_provider == 'anthropic' and bool(reasoning_key),
                'gemini': reasoning_provider == 'gemini' and bool(reasoning_key),
                'elevenlabs': (conversation_config.transcription_provider == 'elevenlabs'
                              or conversation_config.speech_provider == 'elevenlabs') and bool(conversation_config.elevenlabs_api_key),
                'openai_optional': (conversation_config.transcription_provider == 'openai'
                                    or conversation_config.speech_provider == 'openai') and bool(conversation_config.openai_api_key),
            },
        }}

    @api.get('/api/conversation/sessions/{session_id}')
    async def conversation_snapshot(session_id: str, response: Response):
        response.headers['Cache-Control'] = 'no-store'
        return conversation.snapshot(session_id)

    @api.get('/api/fixtures')
    async def fixture_catalog():
        return {'products': fixtures()}

    @api.post('/api/sessions', response_model=Session)
    async def create(request: CreateSession):
        return service.create(request)

    @api.get('/api/sessions/{session_id}', response_model=Session)
    async def session(session_id: str):
        return service.get(session_id).session.model_copy(deep=True)

    @api.get('/api/sessions/{session_id}/latest', response_model=TurnResponse)
    async def latest(session_id: str):
        value = service.get(session_id).latest
        if value is None:
            raise HTTPException(404, 'No completed response in the current session context.')
        return value.model_copy(deep=True)

    @api.get('/api/sessions/{session_id}/frame', response_model=CapturedFrame)
    async def frame(session_id: str, response: Response):
        response.headers['Cache-Control'] = 'no-store'
        return service.get_frame(session_id)

    @api.post('/api/sessions/{session_id}/turn', response_model=TurnResponse)
    async def turn(session_id: str, request: TurnRequest):
        legacy_turn_allowed(session_id)
        return await service.turn(session_id, request)

    def legacy_turn_allowed(session_id: str):
        service.get(session_id)
        runtime = conversation.runtimes.get(session_id)
        if runtime is not None and runtime.connection is not None:
            raise HTTPException(409, 'End the active voice conversation before using turn, label review, or confirmation controls.')

    @api.post('/api/sessions/{session_id}/context', response_model=Session)
    async def context(session_id: str, request: ContextRequest):
        await conversation.interrupt_session(session_id, 'context_changed', recipient_id=request.recipient_id)
        return service.context(session_id, request)

    @api.post('/api/sessions/{session_id}/cancel', response_model=Session)
    async def cancel(session_id: str):
        await conversation.interrupt_session(session_id, 'cancelled')
        return service.cancel(session_id)

    @api.post('/api/sessions/{session_id}/confirm', response_model=TurnResponse)
    async def confirm(session_id: str, request: ConfirmRequest):
        legacy_turn_allowed(session_id)
        return service.confirm(session_id, request.action_id)

    @api.post('/api/sessions/{session_id}/review', response_model=TurnResponse)
    async def review(session_id: str, request: ReviewRequest):
        legacy_turn_allowed(session_id)
        return service.review(session_id, request)

    @api.delete('/api/sessions/{session_id}')
    async def delete(session_id: str):
        await conversation.interrupt_session(session_id, 'deleted', delete=True)
        service.delete(session_id)
        return {'deleted': True}

    return api


app = create_app()
