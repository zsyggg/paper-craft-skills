import { chmod, mkdir, readFile, writeFile } from 'node:fs/promises';
import path from 'node:path';
import { resolveGeminiWebCookiePath } from './paths.js';

export type GeminiWebLog = (message: string) => void;

export const GEMINI_COOKIE_NAMES = [
  '__Secure-1PSID',
  '__Secure-1PSIDTS',
  '__Secure-1PSIDCC',
  '__Secure-1PAPISID',
  'NID',
  'AEC',
  'SOCS',
  '__Secure-BUCKET',
  '__Secure-ENID',
  'SID',
  'HSID',
  'SSID',
  'APISID',
  'SAPISID',
  '__Secure-3PSID',
  '__Secure-3PSIDTS',
  '__Secure-3PAPISID',
  'SIDCC',
] as const;

export const GEMINI_REQUIRED_COOKIES = ['__Secure-1PSID', '__Secure-1PSIDTS'] as const;

export interface GeminiCookieFileV1 {
  version: 1;
  updatedAt: string;
  cookieMap: Record<string, string>;
}

export function hasRequiredGeminiCookies(cookieMap: Record<string, string>): boolean {
  return GEMINI_REQUIRED_COOKIES.every((name) => Boolean(cookieMap[name]));
}

function resolveCookieDomain(cookie: { domain?: string; url?: string }): string | null {
  const rawDomain = cookie.domain?.trim();
  if (rawDomain) {
    return rawDomain.startsWith('.') ? rawDomain.slice(1) : rawDomain;
  }
  const rawUrl = cookie.url?.trim();
  if (rawUrl) {
    try {
      return new URL(rawUrl).hostname;
    } catch {
      return null;
    }
  }
  return null;
}

function pickCookieValue<T extends { name?: string; value?: string; domain?: string; path?: string; url?: string }>(
  cookies: T[],
  name: string,
): string | undefined {
  const matches = cookies.filter((cookie) => cookie.name === name && typeof cookie.value === 'string');
  if (matches.length === 0) return undefined;

  const preferredDomain = matches.find((cookie) => {
    const domain = resolveCookieDomain(cookie);
    return domain === 'google.com' && (cookie.path ?? '/') === '/';
  });
  const googleDomain = matches.find((cookie) => (resolveCookieDomain(cookie) ?? '').endsWith('google.com'));
  return (preferredDomain ?? googleDomain ?? matches[0])?.value;
}

export function buildGeminiCookieMap<
  T extends { name?: string; value?: string; domain?: string; path?: string; url?: string },
>(cookies: T[]): Record<string, string> {
  const cookieMap: Record<string, string> = {};
  for (const name of GEMINI_COOKIE_NAMES) {
    const value = pickCookieValue(cookies, name);
    if (value) cookieMap[name] = value;
  }
  return cookieMap;
}

export async function readGeminiCookieMapFromDisk(options?: {
  cookiePath?: string;
  log?: GeminiWebLog;
}): Promise<Record<string, string>> {
  const cookiePath = options?.cookiePath ?? resolveGeminiWebCookiePath();

  try {
    const raw = await readFile(cookiePath, 'utf8');
    const parsed = JSON.parse(raw) as Partial<GeminiCookieFileV1> | Record<string, unknown>;

    const cookieMap =
      (parsed as Partial<GeminiCookieFileV1>).version === 1
        ? (parsed as Partial<GeminiCookieFileV1>).cookieMap
        : (parsed as Record<string, unknown>);

    if (!cookieMap || typeof cookieMap !== 'object') return {};
    const normalized: Record<string, string> = {};
    for (const [key, value] of Object.entries(cookieMap)) {
      if (typeof value === 'string' && value.trim()) {
        normalized[key] = value;
      }
    }

    if (Object.keys(normalized).length > 0) {
      options?.log?.(`[gemini-web] Loaded cookies from ${cookiePath}`);
    }

    return normalized;
  } catch (error) {
    const code = (error as NodeJS.ErrnoException | undefined)?.code;
    if (code === 'ENOENT') return {};
    options?.log?.(
      `[gemini-web] Failed to read cookies from ${cookiePath}: ${error instanceof Error ? error.message : String(error)}`,
    );
    return {};
  }
}

export async function writeGeminiCookieMapToDisk(
  cookieMap: Record<string, string>,
  options?: { cookiePath?: string; log?: GeminiWebLog },
): Promise<void> {
  const cookiePath = options?.cookiePath ?? resolveGeminiWebCookiePath();
  await mkdir(path.dirname(cookiePath), { recursive: true });

  const payload: GeminiCookieFileV1 = {
    version: 1,
    updatedAt: new Date().toISOString(),
    cookieMap,
  };

  await writeFile(cookiePath, `${JSON.stringify(payload, null, 2)}\n`, { mode: 0o600 });
  try {
    await chmod(cookiePath, 0o600);
  } catch {
    // ignore chmod failures (e.g. on Windows)
  }
  options?.log?.(`[gemini-web] Saved cookies to ${cookiePath}`);
}
