import { mkdir, readFile, writeFile } from 'node:fs/promises';
import path from 'node:path';

export type GeminiWebModelId = 'gemini-3-pro' | 'gemini-2.5-pro' | 'gemini-2.5-flash';

export interface GeminiWebRunInput {
  prompt: string;
  files?: string[];
  model: GeminiWebModelId;
  cookieMap: Record<string, string>;
  chatMetadata?: unknown;
  signal?: AbortSignal;
}

export interface GeminiWebCandidateImage {
  url: string;
  title?: string;
  alt?: string;
  kind: 'web' | 'generated' | 'raw';
}

export interface GeminiWebRunOutput {
  rawResponseText: string;
  text: string;
  thoughts: string | null;
  metadata: unknown;
  images: GeminiWebCandidateImage[];
  errorCode?: number;
  errorMessage?: string;
}

const USER_AGENT =
  'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36';

const MODEL_HEADER_NAME = 'x-goog-ext-525001261-jspb';
const MODEL_HEADERS: Record<GeminiWebModelId, string> = {
  'gemini-3-pro': '[1,null,null,null,"9d8ca3786ebdfbea",null,null,0,[4]]',
  'gemini-2.5-pro': '[1,null,null,null,"4af6c7f5da75d65d",null,null,0,[4]]',
  'gemini-2.5-flash': '[1,null,null,null,"9ec249fc9ad08861",null,null,0,[4]]',
};

const GEMINI_APP_URL = 'https://gemini.google.com/app';
const GEMINI_STREAM_GENERATE_URL =
  'https://gemini.google.com/_/BardChatUi/data/assistant.lamda.BardFrontendService/StreamGenerate';
const GEMINI_UPLOAD_URL = 'https://content-push.googleapis.com/upload';
const GEMINI_UPLOAD_PUSH_ID = 'feeds/mcudyrk2a4khkz';

function getNestedValue<T>(value: unknown, pathParts: Array<string | number>, fallback: T): T {
  let current: unknown = value;
  for (const part of pathParts) {
    if (current == null) return fallback;
    if (typeof part === 'number') {
      if (!Array.isArray(current)) return fallback;
      current = current[part];
    } else {
      if (typeof current !== 'object') return fallback;
      current = (current as Record<string, unknown>)[part];
    }
  }
  return (current as T) ?? fallback;
}

function buildCookieHeader(cookieMap: Record<string, string>): string {
  return Object.entries(cookieMap)
    .filter(([, value]) => typeof value === 'string' && value.length > 0)
    .map(([name, value]) => `${name}=${value}`)
    .join('; ');
}

function getSetCookieHeaders(res: Response): string[] {
  const headers = res.headers as unknown as { getSetCookie?: () => string[] };
  if (typeof headers.getSetCookie === 'function') {
    try {
      return headers.getSetCookie();
    } catch {
      return [];
    }
  }
  const raw = res.headers.get('set-cookie');
  return raw ? [raw] : [];
}

function applySetCookiesToMap(setCookies: string[], cookieMap: Record<string, string>): void {
  for (const raw of setCookies) {
    const first = raw.split(';')[0]?.trim();
    if (!first) continue;
    const idx = first.indexOf('=');
    if (idx <= 0) continue;
    const name = first.slice(0, idx).trim();
    const value = first.slice(idx + 1).trim();
    if (!name) continue;
    cookieMap[name] = value;
  }
}

async function fetchWithCookieJar(
  url: string,
  init: Omit<RequestInit, 'redirect' | 'headers'> & { headers?: Record<string, string> },
  cookieMap: Record<string, string>,
  signal?: AbortSignal,
  maxRedirects = 20,
): Promise<Response> {
  let current = url;
  for (let i = 0; i <= maxRedirects; i += 1) {
    const cookieHeader = buildCookieHeader(cookieMap);
    const headers: Record<string, string> = {
      ...(init.headers ?? {}),
      ...(cookieHeader ? { cookie: cookieHeader } : {}),
      'user-agent': USER_AGENT,
    };

    const res = await fetch(current, { ...init, redirect: 'manual', signal, headers });
    applySetCookiesToMap(getSetCookieHeaders(res), cookieMap);

    if (res.status >= 300 && res.status < 400) {
      const location = res.headers.get('location');
      if (!location) return res;
      current = new URL(location, current).toString();
      continue;
    }

    return res;
  }

  throw new Error(`Too many redirects while fetching ${url} (>${maxRedirects}).`);
}

export async function fetchGeminiAccessToken(
  cookieMap: Record<string, string>,
  signal?: AbortSignal,
): Promise<string> {
  const res = await fetchWithCookieJar(GEMINI_APP_URL, { method: 'GET' }, cookieMap, signal);
  const html = await res.text();

  const tokens = ['SNlM0e', 'thykhd'] as const;
  for (const key of tokens) {
    const match = html.match(new RegExp(`"${key}":"(.*?)"`));
    if (match?.[1]) return match[1];
  }
  throw new Error(
    'Unable to locate Gemini access token on gemini.google.com/app (missing SNlM0e/thykhd).',
  );
}

function trimGeminiJsonEnvelope(text: string): string {
  const start = text.indexOf('[');
  const end = text.lastIndexOf(']');
  if (start === -1 || end === -1 || end <= start) {
    throw new Error('Gemini response did not contain a JSON payload.');
  }
  return text.slice(start, end + 1);
}

function extractErrorCode(responseJson: unknown): number | undefined {
  const code = getNestedValue<number>(responseJson, [0, 5, 2, 0, 1, 0], -1);
  return typeof code === 'number' && code >= 0 ? code : undefined;
}

function extractGgdlUrls(rawText: string): string[] {
  const matches =
    rawText.match(/https?:\/\/[^/\s"']*googleusercontent\.com\/gg-dl\/[^\s"']+/g) ?? [];
  const seen = new Set<string>();
  const urls: string[] = [];
  for (const match of matches) {
    if (seen.has(match)) continue;
    seen.add(match);
    urls.push(match);
  }
  return urls;
}

function extractImageGenerationContentUrls(rawText: string): string[] {
  const matches =
    rawText.match(/https?:\/\/googleusercontent\.com\/image_generation_content\/\d+/g) ?? [];
  const seen = new Set<string>();
  const urls: string[] = [];
  for (const match of matches) {
    if (seen.has(match)) continue;
    seen.add(match);
    urls.push(match);
  }
  return urls;
}

function ensureFullSizeImageUrl(url: string): string {
  const trimmed = url.trim();
  let normalized = trimmed;
  const backslashIndex = normalized.indexOf('\\');
  if (backslashIndex >= 0) normalized = normalized.slice(0, backslashIndex);
  // Some Gemini responses embed a size suffix as "/=s2048" which breaks downloads.
  normalized = normalized.replace(/\/=s(?=\d+(?:$|[?#]))/, '=s');
  normalized = normalized.replace(/\/=s(?=$|[?#])/, '=s');
  if (normalized.endsWith('/')) normalized = normalized.slice(0, -1);
  if (normalized.includes('=s2048')) return normalized;
  if (normalized.includes('=s')) return normalized;
  return `${normalized}=s2048`;
}

async function fetchWithCookiePreservingRedirects(
  url: string,
  init: Omit<RequestInit, 'redirect'>,
  signal?: AbortSignal,
  maxRedirects = 10,
): Promise<Response> {
  let current = url;
  for (let i = 0; i <= maxRedirects; i += 1) {
    const res = await fetch(current, { ...init, redirect: 'manual', signal });
    if (res.status >= 300 && res.status < 400) {
      const location = res.headers.get('location');
      if (!location) return res;
      current = new URL(location, current).toString();
      continue;
    }
    return res;
  }
  throw new Error(`Too many redirects while downloading image (>${maxRedirects}).`);
}

export async function downloadGeminiImage(
  url: string,
  cookieMap: Record<string, string>,
  outputPath: string,
  signal?: AbortSignal,
): Promise<void> {
  const cookieHeader = buildCookieHeader(cookieMap);
  const res = await fetchWithCookiePreservingRedirects(ensureFullSizeImageUrl(url), {
    headers: {
      cookie: cookieHeader,
      'user-agent': USER_AGENT,
    },
  }, signal);
  if (!res.ok) {
    throw new Error(`Failed to download image: ${res.status} ${res.statusText} (${res.url})`);
  }

  const data = new Uint8Array(await res.arrayBuffer());
  await mkdir(path.dirname(outputPath), { recursive: true });
  await writeFile(outputPath, data);
}

async function uploadGeminiFile(filePath: string, signal?: AbortSignal): Promise<{ id: string; name: string }> {
  const absPath = path.resolve(process.cwd(), filePath);
  const data = await readFile(absPath);
  const fileName = path.basename(absPath);
  const form = new FormData();
  form.append('file', new Blob([data]), fileName);

  const res = await fetch(GEMINI_UPLOAD_URL, {
    method: 'POST',
    redirect: 'follow',
    signal,
    headers: {
      'push-id': GEMINI_UPLOAD_PUSH_ID,
      'user-agent': USER_AGENT,
    },
    body: form,
  });
  const text = await res.text();
  if (!res.ok) {
    throw new Error(`File upload failed: ${res.status} ${res.statusText} (${text.slice(0, 200)})`);
  }
  return { id: text, name: fileName };
}

function guessMimeType(fileName: string): string {
  const ext = path.extname(fileName).toLowerCase();
  switch (ext) {
    case '.png':
      return 'image/png';
    case '.jpg':
    case '.jpeg':
      return 'image/jpeg';
    case '.webp':
      return 'image/webp';
    case '.gif':
      return 'image/gif';
    case '.mp4':
      return 'video/mp4';
    case '.mov':
      return 'video/quicktime';
    case '.webm':
      return 'video/webm';
    default:
      return 'application/octet-stream';
  }
}

function buildGeminiFReqPayload(
  prompt: string,
  uploaded: Array<{ id: string; name: string }>,
  chatMetadata: unknown,
): string {
  const promptPayload =
    uploaded.length > 0
      ? [
          prompt,
          0,
          null,
          // Matches gemini-web payload format: [[[fileId, 1, null, mimeType], fileName]] for an attachment.
          uploaded.map((file) => [[file.id, 1, null, guessMimeType(file.name)], file.name]),
        ]
      : [prompt];

  const innerList: unknown[] = [promptPayload, null, chatMetadata ?? null];
  return JSON.stringify([null, JSON.stringify(innerList)]);
}

export function parseGeminiStreamGenerateResponse(rawText: string): {
  metadata: unknown;
  text: string;
  thoughts: string | null;
  images: GeminiWebCandidateImage[];
  errorCode?: number;
} {
  const responseJson = JSON.parse(trimGeminiJsonEnvelope(rawText)) as unknown;
  const errorCode = extractErrorCode(responseJson);

  const parts = Array.isArray(responseJson) ? responseJson : [];
  let bodyIndex = 0;
  let body: unknown = null;
  for (let i = 0; i < parts.length; i += 1) {
    const partBody = getNestedValue<string | null>(parts[i], [2], null);
    if (!partBody) continue;
    try {
      const parsed = JSON.parse(partBody) as unknown;
      const candidateList = getNestedValue<unknown[]>(parsed, [4], []);
      if (Array.isArray(candidateList) && candidateList.length > 0) {
        bodyIndex = i;
        body = parsed;
        break;
      }
    } catch {
      // ignore
    }
  }

  const candidateList = getNestedValue<unknown[]>(body, [4], []);
  const firstCandidate = candidateList[0];
  const textRaw = getNestedValue<string>(firstCandidate, [1, 0], '');
  const cardContent = /^http:\/\/googleusercontent\.com\/card_content\/\d+/.test(textRaw);
  const text = cardContent
    ? (getNestedValue<string | null>(firstCandidate, [22, 0], null) ?? textRaw)
    : textRaw;
  const thoughts = getNestedValue<string | null>(firstCandidate, [37, 0, 0], null);
  const conversationMeta = getNestedValue<unknown[]>(body, [1], []);
  const conversationId =
    typeof conversationMeta[0] === 'string' && conversationMeta[0].length > 0
      ? conversationMeta[0]
      : null;
  const responseId =
    typeof conversationMeta[1] === 'string' && conversationMeta[1].length > 0
      ? conversationMeta[1]
      : null;
  const choiceIdRaw = getNestedValue<string | null>(firstCandidate, [0], null);
  const choiceId = typeof choiceIdRaw === 'string' && choiceIdRaw.length > 0 ? choiceIdRaw : null;
  const metadata =
    conversationId && responseId && choiceId ? [conversationId, responseId, choiceId] : conversationMeta;

  const images: GeminiWebCandidateImage[] = [];

  const webImages = getNestedValue<unknown[]>(firstCandidate, [12, 1], []);
  for (const webImage of webImages) {
    const url = getNestedValue<string | null>(webImage, [0, 0, 0], null);
    if (!url) continue;
    images.push({
      kind: 'web',
      url,
      title: getNestedValue<string | undefined>(webImage, [7, 0], undefined),
      alt: getNestedValue<string | undefined>(webImage, [0, 4], undefined),
    });
  }

  const hasGenerated = Boolean(getNestedValue<unknown>(firstCandidate, [12, 7, 0], null));
  if (hasGenerated) {
    let imgBody: unknown = null;
    for (let i = bodyIndex; i < parts.length; i += 1) {
      const partBody = getNestedValue<string | null>(parts[i], [2], null);
      if (!partBody) continue;
      try {
        const parsed = JSON.parse(partBody) as unknown;
        const candidateImages = getNestedValue<unknown | null>(parsed, [4, 0, 12, 7, 0], null);
        if (candidateImages != null) {
          imgBody = parsed;
          break;
        }
      } catch {
        // ignore
      }
    }

    const imgCandidate = getNestedValue<unknown>(imgBody ?? body, [4, 0], null);

    const generated = getNestedValue<unknown[]>(imgCandidate, [12, 7, 0], []);
    for (const genImage of generated) {
      const url = getNestedValue<string | null>(genImage, [0, 3, 3], null);
      if (!url) continue;
      images.push({
        kind: 'generated',
        url,
        title: '[Generated Image]',
        alt: '',
      });
    }
  }

  return { metadata, text, thoughts, images, errorCode };
}

export function isGeminiModelUnavailable(errorCode: number | undefined): boolean {
  return errorCode === 1052;
}

export async function runGeminiWebOnce(input: GeminiWebRunInput): Promise<GeminiWebRunOutput> {
  const at = await fetchGeminiAccessToken(input.cookieMap, input.signal);
  const cookieHeader = buildCookieHeader(input.cookieMap);

  const uploaded: Array<{ id: string; name: string }> = [];
  for (const file of input.files ?? []) {
    if (input.signal?.aborted) {
      throw new Error('Gemini web run aborted before upload.');
    }
    uploaded.push(await uploadGeminiFile(file, input.signal));
  }

  const fReq = buildGeminiFReqPayload(input.prompt, uploaded, input.chatMetadata ?? null);
  const params = new URLSearchParams();
  params.set('at', at);
  params.set('f.req', fReq);

  const res = await fetch(GEMINI_STREAM_GENERATE_URL, {
    method: 'POST',
    redirect: 'follow',
    signal: input.signal,
    headers: {
      'content-type': 'application/x-www-form-urlencoded;charset=utf-8',
      origin: 'https://gemini.google.com',
      referer: 'https://gemini.google.com/',
      'x-same-domain': '1',
      'user-agent': USER_AGENT,
      cookie: cookieHeader,
      [MODEL_HEADER_NAME]: MODEL_HEADERS[input.model],
    },
    body: params.toString(),
  });

  const rawResponseText = await res.text();
  if (!res.ok) {
    return {
      rawResponseText,
      text: '',
      thoughts: null,
      metadata: input.chatMetadata ?? null,
      images: [],
      errorMessage: `Gemini request failed: ${res.status} ${res.statusText}`,
    };
  }

  try {
    const parsed = parseGeminiStreamGenerateResponse(rawResponseText);
    return {
      rawResponseText,
      text: parsed.text ?? '',
      thoughts: parsed.thoughts,
      metadata: parsed.metadata,
      images: parsed.images,
      errorCode: parsed.errorCode,
    };
  } catch (error) {
    let responseJson: unknown = null;
    try {
      responseJson = JSON.parse(trimGeminiJsonEnvelope(rawResponseText)) as unknown;
    } catch {
      responseJson = null;
    }
    const errorCode = extractErrorCode(responseJson);

    return {
      rawResponseText,
      text: '',
      thoughts: null,
      metadata: input.chatMetadata ?? null,
      images: [],
      errorCode: typeof errorCode === 'number' ? errorCode : undefined,
      errorMessage: error instanceof Error ? error.message : String(error ?? ''),
    };
  }
}

export async function runGeminiWebWithFallback(
  input: Omit<GeminiWebRunInput, 'model'> & { model: GeminiWebModelId },
): Promise<GeminiWebRunOutput & { effectiveModel: GeminiWebModelId }> {
  const attempt = await runGeminiWebOnce(input);
  if (isGeminiModelUnavailable(attempt.errorCode) && input.model !== 'gemini-2.5-flash') {
    const fallback = await runGeminiWebOnce({ ...input, model: 'gemini-2.5-flash' });
    return { ...fallback, effectiveModel: 'gemini-2.5-flash' };
  }
  return { ...attempt, effectiveModel: input.model };
}

export async function saveFirstGeminiImageFromOutput(
  output: GeminiWebRunOutput,
  cookieMap: Record<string, string>,
  outputPath: string,
  signal?: AbortSignal,
): Promise<{ saved: boolean; imageCount: number }> {
  const generatedOrWeb = output.images.find((img) => img.kind === 'generated') ?? output.images[0];
  if (generatedOrWeb?.url) {
    await downloadGeminiImage(generatedOrWeb.url, cookieMap, outputPath, signal);
    return { saved: true, imageCount: output.images.length };
  }

  const ggdl = extractGgdlUrls(`${output.text}\n${output.rawResponseText}`);
  const preferred = ggdl.length > 0 ? ggdl[ggdl.length - 1] : null;
  if (preferred) {
    await downloadGeminiImage(preferred, cookieMap, outputPath, signal);
    return { saved: true, imageCount: ggdl.length };
  }

  const imageGen = extractImageGenerationContentUrls(`${output.text}\n${output.rawResponseText}`);
  const imageGenPreferred = imageGen.length > 0 ? imageGen[imageGen.length - 1] : null;
  if (imageGenPreferred) {
    await downloadGeminiImage(imageGenPreferred, cookieMap, outputPath, signal);
    return { saved: true, imageCount: imageGen.length };
  }

  return { saved: false, imageCount: 0 };
}
