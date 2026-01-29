import { mkdir, writeFile } from 'node:fs/promises';
import path from 'node:path';
import type { BrowserRunOptions, BrowserRunResult, BrowserLogger, CookieParam } from '../browser/types.js';
import { runGeminiWebWithFallback, saveFirstGeminiImageFromOutput } from './client.js';
import type { GeminiWebModelId, GeminiWebRunOutput } from './client.js';
import {
  buildGeminiCookieMap,
  hasRequiredGeminiCookies,
  readGeminiCookieMapFromDisk,
} from './cookie-store.js';
import type { GeminiWebOptions, GeminiWebResponse } from './types.js';

export { hasRequiredGeminiCookies } from './cookie-store.js';

const USER_AGENT =
  'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36';

function estimateTokenCount(text: string): number {
  return Math.ceil(text.length / 4);
}

function resolveInvocationPath(value: string | undefined): string | undefined {
  if (!value) return undefined;
  const trimmed = value.trim();
  if (!trimmed) return undefined;
  return path.isAbsolute(trimmed) ? trimmed : path.resolve(process.cwd(), trimmed);
}

function normalizePathList(value: string | string[] | undefined): string[] {
  if (!value) return [];
  const raw = Array.isArray(value) ? value : [value];
  const out: string[] = [];
  for (const entry of raw) {
    if (typeof entry !== 'string') continue;
    const resolved = resolveInvocationPath(entry);
    if (!resolved) continue;
    out.push(resolved);
  }
  return out;
}

function dedupePaths(paths: string[]): string[] {
  const seen = new Set<string>();
  const out: string[] = [];
  for (const item of paths) {
    const trimmed = item.trim();
    if (!trimmed || seen.has(trimmed)) continue;
    seen.add(trimmed);
    out.push(trimmed);
  }
  return out;
}

function buildCookieHeader(cookieMap: Record<string, string>): string {
  return Object.entries(cookieMap)
    .filter(([, value]) => typeof value === 'string' && value.length > 0)
    .map(([name, value]) => `${name}=${value}`)
    .join('; ');
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
  throw new Error(`Too many redirects while downloading media (>${maxRedirects}).`);
}

async function downloadGeminiMedia(
  url: string,
  cookieMap: Record<string, string>,
  outputPath: string,
  signal?: AbortSignal,
): Promise<void> {
  const cookieHeader = buildCookieHeader(cookieMap);
  const res = await fetchWithCookiePreservingRedirects(
    url,
    {
      headers: {
        cookie: cookieHeader,
        'user-agent': USER_AGENT,
      },
    },
    signal,
  );
  if (!res.ok) {
    throw new Error(`Failed to download media: ${res.status} ${res.statusText} (${res.url})`);
  }

  const data = new Uint8Array(await res.arrayBuffer());
  await mkdir(path.dirname(outputPath), { recursive: true });
  await writeFile(outputPath, data);
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

async function saveFirstGeminiVideoFromOutput(
  output: GeminiWebRunOutput,
  cookieMap: Record<string, string>,
  outputPath: string,
  signal?: AbortSignal,
): Promise<{ saved: boolean; videoCount: number }> {
  const ggdl = extractGgdlUrls(output.rawResponseText);
  if (!ggdl[0]) return { saved: false, videoCount: 0 };

  const videoCandidates = ggdl.filter((url) => /\.(mp4|webm|mov)(?:$|[?#])/i.test(url));
  const preferred =
    (videoCandidates.length > 0 ? videoCandidates[videoCandidates.length - 1] : null) ??
    ggdl.find((url) => /video/i.test(url)) ??
    ggdl[ggdl.length - 1];
  await downloadGeminiMedia(preferred, cookieMap, outputPath, signal);
  return { saved: true, videoCount: ggdl.length };
}

function resolveGeminiWebModel(
  desiredModel: string | null | undefined,
  log?: BrowserLogger,
): GeminiWebModelId {
  const desired = typeof desiredModel === 'string' ? desiredModel.trim() : '';
  if (!desired) return 'gemini-3-pro';

  switch (desired) {
    case 'gemini-3-pro':
    case 'gemini-3.0-pro':
      return 'gemini-3-pro';
    case 'gemini-2.5-pro':
      return 'gemini-2.5-pro';
    case 'gemini-2.5-flash':
      return 'gemini-2.5-flash';
    default:
      if (desired.startsWith('gemini-')) {
        log?.(
          `[gemini-web] Unsupported Gemini web model "${desired}". Falling back to gemini-3-pro.`,
        );
      }
      return 'gemini-3-pro';
  }
}

function buildInlineCookiesFromEnv(): CookieParam[] {
  const cookies: CookieParam[] = [];
  const psid = process.env.GEMINI_SECURE_1PSID?.trim();
  const psidts = process.env.GEMINI_SECURE_1PSIDTS?.trim();

  if (psid) {
    cookies.push({ name: '__Secure-1PSID', value: psid, domain: 'google.com', path: '/' });
  }
  if (psidts) {
    cookies.push({ name: '__Secure-1PSIDTS', value: psidts, domain: 'google.com', path: '/' });
  }

  return cookies;
}

async function loadGeminiCookiesFromInline(
  browserConfig: BrowserRunOptions['config'],
  log?: BrowserLogger,
): Promise<Record<string, string>> {
  const inline = browserConfig?.inlineCookies;
  if (!inline || inline.length === 0) return {};

  const cookieMap = buildGeminiCookieMap(
    inline.filter((cookie): cookie is CookieParam => Boolean(cookie?.name && typeof cookie.value === 'string')),
  );

  if (Object.keys(cookieMap).length > 0) {
    const source = browserConfig?.inlineCookiesSource ?? 'inline';
    log?.(`[gemini-web] Loaded Gemini cookies from inline payload (${source}): ${Object.keys(cookieMap).length} cookie(s).`);
  } else {
    log?.('[gemini-web] Inline cookie payload provided but no Gemini cookies matched.');
  }

  return cookieMap;
}

export async function loadGeminiCookies(
  browserConfig: BrowserRunOptions['config'],
  log?: BrowserLogger,
): Promise<Record<string, string>> {
  const inlineMap = await loadGeminiCookiesFromInline(browserConfig, log);
  if (hasRequiredGeminiCookies(inlineMap)) return inlineMap;

  const diskMap = await readGeminiCookieMapFromDisk({ log });
  const merged = { ...diskMap, ...inlineMap };
  if (hasRequiredGeminiCookies(merged)) return merged;

  if (browserConfig?.cookieSync === false) {
    log?.('[gemini-web] Cookie sync disabled and inline cookies missing Gemini auth tokens.');
    return merged;
  }

  log?.(
    '[gemini-web] Missing Gemini auth cookies. Run `npx -y bun skills/baoyu-gemini-web/scripts/main.ts --login` to sign in and refresh cookies.',
  );
  return merged;
}

export async function loadGeminiCookieMap(log?: BrowserLogger): Promise<Record<string, string>> {
  const diskMap = await readGeminiCookieMapFromDisk({ log });
  const inlineCookies = buildInlineCookiesFromEnv();
  const envMap = buildGeminiCookieMap(inlineCookies);
  return { ...diskMap, ...envMap };
}

export function createGeminiWebExecutor(
  geminiOptions: GeminiWebOptions,
): (runOptions: BrowserRunOptions) => Promise<BrowserRunResult> {
  let persistedChatMetadata: unknown | null = null;
  let referenceImagesUploaded = false;

  return async (runOptions: BrowserRunOptions): Promise<BrowserRunResult> => {
    const startTime = Date.now();
    const log = runOptions.log;

    log?.('[gemini-web] Starting Gemini web executor (TypeScript)');

    const cookieMap = await loadGeminiCookies(runOptions.config, log);
    if (!hasRequiredGeminiCookies(cookieMap)) {
      throw new Error(
        'Gemini browser mode requires auth cookies (missing __Secure-1PSID/__Secure-1PSIDTS). Run `npx -y bun skills/baoyu-gemini-web/scripts/main.ts --login` to sign in and save cookies.',
      );
    }

    const configTimeout =
      typeof runOptions.config?.timeoutMs === 'number' && Number.isFinite(runOptions.config.timeoutMs)
        ? Math.max(1_000, runOptions.config.timeoutMs)
        : null;

    const generateVideoPath = resolveInvocationPath(geminiOptions.generateVideo);

    const defaultTimeoutMs = geminiOptions.youtube
      ? 240_000
      : generateVideoPath
        ? 900_000
        : geminiOptions.generateImage || geminiOptions.editImage
        ? 300_000
        : 120_000;

    const timeoutCapMs = generateVideoPath ? 1_800_000 : 600_000;
    const timeoutMs = Math.min(configTimeout ?? defaultTimeoutMs, timeoutCapMs);
    const controller = new AbortController();
    const timeout = setTimeout(() => controller.abort(), timeoutMs);

    const keepSession = geminiOptions.keepSession === true;

    const generateImagePath = resolveInvocationPath(geminiOptions.generateImage);
    const editImagePath = resolveInvocationPath(geminiOptions.editImage);
    const outputPath = resolveInvocationPath(geminiOptions.outputPath);
    const attachmentPaths = (runOptions.attachments ?? []).map((attachment) => attachment.path);
    const referenceImagePaths = normalizePathList(geminiOptions.referenceImages);
    const requestFilePaths = dedupePaths(
      keepSession ? attachmentPaths : [...referenceImagePaths, ...attachmentPaths],
    );

    if (generateVideoPath && (generateImagePath || editImagePath)) {
      throw new Error('Gemini web executor: generateVideo cannot be combined with generateImage/editImage options.');
    }

    let prompt = runOptions.prompt;
    if (geminiOptions.aspectRatio && (generateImagePath || editImagePath || generateVideoPath)) {
      prompt = `${prompt} (aspect ratio: ${geminiOptions.aspectRatio})`;
    }
    if (geminiOptions.youtube) {
      prompt = `${prompt}\n\nYouTube video: ${geminiOptions.youtube}`;
    }
    if (generateImagePath && !editImagePath) {
      prompt = `Generate an image: ${prompt}`;
    }
    if (generateVideoPath) {
      prompt = `Generate a video: ${prompt}`;
    }

    const model: GeminiWebModelId = resolveGeminiWebModel(runOptions.config?.desiredModel, log);
    let response: GeminiWebResponse;
    let videoSaveSummary: { saved: boolean; videoCount: number; outputPath: string } | null = null;

    try {
      let chatMetadata: unknown = keepSession ? persistedChatMetadata : null;

      if (keepSession && referenceImagePaths.length > 0 && !referenceImagesUploaded) {
        const intro = await runGeminiWebWithFallback({
          prompt: 'Here are reference images for future messages.',
          files: referenceImagePaths,
          model,
          cookieMap,
          chatMetadata,
          signal: controller.signal,
        });
        chatMetadata = intro.metadata;
        persistedChatMetadata = intro.metadata;
        referenceImagesUploaded = true;
      }

      if (editImagePath) {
        const intro = await runGeminiWebWithFallback({
          prompt: 'Here is an image to edit',
          files: [editImagePath],
          model,
          cookieMap,
          chatMetadata,
          signal: controller.signal,
        });
        const editPrompt = `Use image generation tool to ${prompt}`;
        const out = await runGeminiWebWithFallback({
          prompt: editPrompt,
          files: requestFilePaths,
          model,
          cookieMap,
          chatMetadata: intro.metadata,
          signal: controller.signal,
        });
        if (keepSession) persistedChatMetadata = out.metadata;
        response = {
          text: out.text ?? null,
          thoughts: geminiOptions.showThoughts ? out.thoughts : null,
          has_images: false,
          image_count: 0,
        };

        const resolvedOutputPath = outputPath ?? generateImagePath ?? 'generated.png';
        const imageSave = await saveFirstGeminiImageFromOutput(out, cookieMap, resolvedOutputPath, controller.signal);
        response.has_images = imageSave.saved;
        response.image_count = imageSave.imageCount;
        if (!imageSave.saved) {
          throw new Error(`No images generated. Response text:\n${out.text || '(empty response)'}`);
        }
      } else if (generateImagePath) {
        const out = await runGeminiWebWithFallback({
          prompt,
          files: requestFilePaths,
          model,
          cookieMap,
          chatMetadata,
          signal: controller.signal,
        });
        if (keepSession) persistedChatMetadata = out.metadata;
        response = {
          text: out.text ?? null,
          thoughts: geminiOptions.showThoughts ? out.thoughts : null,
          has_images: false,
          image_count: 0,
        };
        const imageSave = await saveFirstGeminiImageFromOutput(out, cookieMap, generateImagePath, controller.signal);
        response.has_images = imageSave.saved;
        response.image_count = imageSave.imageCount;
        if (!imageSave.saved) {
          throw new Error(`No images generated. Response text:\n${out.text || '(empty response)'}`);
        }
      } else if (generateVideoPath) {
        const out = await runGeminiWebWithFallback({
          prompt,
          files: requestFilePaths,
          model,
          cookieMap,
          chatMetadata,
          signal: controller.signal,
        });
        if (keepSession) persistedChatMetadata = out.metadata;
        response = {
          text: out.text ?? null,
          thoughts: geminiOptions.showThoughts ? out.thoughts : null,
          has_images: false,
          image_count: 0,
        };

        const resolvedOutputPath = generateVideoPath ?? outputPath ?? 'generated.mp4';
        const save = await saveFirstGeminiVideoFromOutput(out, cookieMap, resolvedOutputPath, controller.signal);
        videoSaveSummary = { ...save, outputPath: resolvedOutputPath };
      } else {
        const out = await runGeminiWebWithFallback({
          prompt,
          files: requestFilePaths,
          model,
          cookieMap,
          chatMetadata,
          signal: controller.signal,
        });
        if (keepSession) persistedChatMetadata = out.metadata;
        response = {
          text: out.text ?? null,
          thoughts: geminiOptions.showThoughts ? out.thoughts : null,
          has_images: out.images.length > 0,
          image_count: out.images.length,
        };
      }
    } finally {
      clearTimeout(timeout);
    }

    const answerText = response.text ?? '';
    let answerMarkdown = answerText;

    if (geminiOptions.showThoughts && response.thoughts) {
      answerMarkdown = `## Thinking\n\n${response.thoughts}\n\n## Response\n\n${answerText}`;
    }

    if (response.has_images && response.image_count > 0) {
      const imagePath = generateImagePath || outputPath || 'generated.png';
      answerMarkdown += `\n\n*Generated ${response.image_count} image(s). Saved to: ${imagePath}*`;
    }
    if (videoSaveSummary) {
      if (videoSaveSummary.saved) {
        answerMarkdown += `\n\n*Generated ${videoSaveSummary.videoCount || 1} video(s). Saved to: ${videoSaveSummary.outputPath}*`;
      } else if (/video_gen_chip/.test(answerMarkdown) || /video_gen_chip/.test(response.text ?? '')) {
        answerMarkdown += '\n\n*Video generation is asynchronous. Check Gemini web UI to download the result.*';
      } else {
        answerMarkdown += '\n\n*No downloadable video URL found in Gemini response.*';
      }
    }

    const tookMs = Date.now() - startTime;
    log?.(`[gemini-web] Completed in ${tookMs}ms`);

    return {
      answerText,
      answerMarkdown,
      tookMs,
      answerTokens: estimateTokenCount(answerText),
      answerChars: answerText.length,
    };
  };
}
