"""PDF pipeline failures that carry the data recovery needs."""

from __future__ import annotations


class PDFPageRenderError(RuntimeError):
    """A page could not be redrawn safely and may be retried from source."""

    def __init__(
        self,
        page: int,
        element: int,
        reason: str,
        *,
        retryable: bool = True,
    ):
        self.page = int(page)
        self.element = int(element)
        self.retryable = bool(retryable)
        self.fallback_pages = [self.page] if self.page > 0 else []
        self.repair_pages = (
            [self.page] if self.retryable and self.page > 0 else []
        )
        super().__init__(f"Page {self.page} elem {self.element}: {reason}")


class PDFPageExtractionError(RuntimeError):
    """A source page could not be extracted safely and may be retried once."""

    def __init__(self, page: int, reason: str):
        self.page = int(page)
        self.fallback_pages = [self.page] if self.page > 0 else []
        self.repair_pages = [self.page] if self.page > 0 else []
        super().__init__(f"Page {self.page} extraction failed: {reason}")


class PDFTranslationIntegrityError(RuntimeError):
    """A page contains a required element without a verified translation."""

    def __init__(self, page: int, issues: list[dict], translations: dict | None = None):
        self.page = int(page)
        self.issues = list(issues or [])
        # The surviving per-element candidates let the caller downgrade a
        # page-level failure to element-level source fallbacks instead of
        # discarding every verified translation on the page.
        self.translations = dict(translations or {})
        summary = ", ".join(
            f"elem {issue.get('element')} {issue.get('reason')}"
            for issue in self.issues[:5]
        )
        super().__init__(
            f"PDF translation integrity failed on page {self.page}: "
            f"{summary or 'unknown translation defect'}"
        )


class PDFPageTranslationError(RuntimeError):
    """One or more pages failed both the normal and clean translation pass."""

    def __init__(self, errors: dict[int, Exception]):
        self.errors = dict(errors or {})
        self.pages = sorted(int(page) for page in self.errors)
        summary = "; ".join(
            f"page {page}: {self.errors[page]}" for page in self.pages[:5]
        )
        super().__init__(
            f"PDF page translation failed after one clean retry "
            f"({len(self.pages)} page(s)): {summary}"
        )


class PDFStructureValidationError(RuntimeError):
    """A final PDF audit failure with pages eligible for one clean rebuild."""

    def __init__(self, check: dict):
        self.check = dict(check or {})
        status = str(self.check.get("status", "error"))
        warnings = self.check.get("warnings") or []
        try:
            warning_count = int(self.check.get("warning_count", len(warnings)))
        except (TypeError, ValueError):
            warning_count = max(1, len(warnings))
        warning_types = sorted({
            str(warning.get("type", "unknown"))
            for warning in warnings
            if isinstance(warning, dict)
        })
        summary = ", ".join(warning_types[:5]) or status
        self.repair_pages = sorted({
            int(warning["page"])
            for warning in warnings
            if isinstance(warning, dict)
            and str(warning.get("page", "")).isdigit()
            and int(warning["page"]) > 0
        })
        super().__init__(
            f"PDF structure validation failed ({warning_count} warning(s): {summary})"
        )
