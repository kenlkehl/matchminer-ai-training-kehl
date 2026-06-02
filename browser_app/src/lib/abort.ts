export function abortError(message = "Stopped by user"): Error {
  if (typeof DOMException !== "undefined") {
    return new DOMException(message, "AbortError");
  }
  const error = new Error(message);
  error.name = "AbortError";
  return error;
}

export function assertNotAborted(signal?: AbortSignal | null): void {
  if (signal?.aborted) throw abortError();
}

export function isAbortError(error: unknown): boolean {
  return typeof error === "object" && error !== null && (error as { name?: unknown }).name === "AbortError";
}
