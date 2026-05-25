/**
 * Type declarations for `wav-decoder` (mohayonao/wav-decoder).
 *
 * This package is pure JS with no bundled types and no @types/ package.
 * These declarations cover the subset of the API used by audiovec.
 */

declare module "wav-decoder" {
  export interface AudioData {
    sampleRate: number;
    channelData: Float32Array[];
    length: number;
    bitsPerSample: number;
  }

  export function decode(
    buffer: ArrayBuffer | Buffer | Uint8Array
  ): Promise<AudioData>;
}
