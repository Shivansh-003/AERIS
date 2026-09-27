from backend.app.schemas.error import InvalidParam, ProblemDetails
from fastapi import FastAPI, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException


class AERISException(Exception):
    def __init__(
        self,
        title: str = "AERIS Application Error",
        detail: str = "An unexpected error occurred within the AERIS platform.",
        status_code: int = status.HTTP_500_INTERNAL_SERVER_ERROR,
        error_type: str = "https://aeris.ai/errors/internal-error",
    ):
        self.title = title
        self.detail = detail
        self.status_code = status_code
        self.error_type = error_type
        super().__init__(detail)


def register_exception_handlers(app: FastAPI) -> None:
    @app.exception_handler(AERISException)
    async def aeris_exception_handler(
        request: Request, exc: AERISException
    ) -> JSONResponse:
        problem = ProblemDetails(
            type=exc.error_type,
            title=exc.title,
            status=exc.status_code,
            detail=exc.detail,
            instance=str(request.url.path),
        )
        return JSONResponse(
            status_code=exc.status_code,
            content=problem.model_dump(exclude_none=True),
        )

    @app.exception_handler(StarletteHTTPException)
    async def http_exception_handler(
        request: Request, exc: StarletteHTTPException
    ) -> JSONResponse:
        problem = ProblemDetails(
            type=f"https://aeris.ai/errors/http-{exc.status_code}",
            title=exc.detail if isinstance(exc.detail, str) else "HTTP Error",
            status=exc.status_code,
            detail=str(exc.detail),
            instance=str(request.url.path),
        )
        return JSONResponse(
            status_code=exc.status_code,
            content=problem.model_dump(exclude_none=True),
        )

    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(
        request: Request, exc: RequestValidationError
    ) -> JSONResponse:
        invalid_params = []
        for error in exc.errors():
            loc = " -> ".join([str(x) for x in error.get("loc", [])])
            msg = error.get("msg", "Invalid parameter")
            invalid_params.append(InvalidParam(name=loc, reason=msg))

        problem = ProblemDetails(
            type="https://aeris.ai/errors/validation-error",
            title="Request Validation Error",
            status=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="The request body or parameters failed schema validation.",
            instance=str(request.url.path),
            invalid_params=invalid_params,
        )
        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            content=problem.model_dump(exclude_none=True),
        )
