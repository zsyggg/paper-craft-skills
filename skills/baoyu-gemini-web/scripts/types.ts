export interface GeminiWebOptions {
  youtube?: string;
  generateImage?: string;
  editImage?: string;
  generateVideo?: string;
  outputPath?: string;
  showThoughts?: boolean;
  aspectRatio?: string;
  /**
   * One or more local image paths to upload as persistent reference images.
   * - If `keepSession` is enabled, they are uploaded once per executor session.
   * - Otherwise, they are attached to each request.
   */
  referenceImages?: string | string[];
  /** Preserve Gemini chat metadata to continue multi-turn conversations within the same executor instance. */
  keepSession?: boolean;
}

export interface GeminiWebResponse {
  text: string | null;
  thoughts: string | null;
  has_images: boolean;
  image_count: number;
  error?: string;
}
