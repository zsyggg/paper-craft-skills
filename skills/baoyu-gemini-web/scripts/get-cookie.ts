#!/usr/bin/env -S npx -y bun

import process from 'node:process';

import { getGeminiCookieMapViaChrome } from './chrome-auth.js';
import { writeGeminiCookieMapToDisk } from './cookie-store.js';
import { resolveGeminiWebChromeProfileDir, resolveGeminiWebCookiePath } from './paths.js';

async function main(): Promise<void> {
  const cookiePath = resolveGeminiWebCookiePath();
  const profileDir = resolveGeminiWebChromeProfileDir();

  const log = (msg: string) => console.log(msg);
  const cookieMap = await getGeminiCookieMapViaChrome({ userDataDir: profileDir, log });
  await writeGeminiCookieMapToDisk(cookieMap, { cookiePath, log });
}

main().catch((error) => {
  console.error(error instanceof Error ? error.message : String(error));
  process.exit(1);
});
