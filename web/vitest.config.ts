import { defineConfig } from "vitest/config";
import path from "path";

export default defineConfig({
  test: {
    globals: true,
    environment: "node",
    include: ["lib/**/*.test.ts"],
    exclude: ["node_modules", ".next"],
    // Don't auto-reset mocks — we manage state manually in beforeEach
    // to preserve module-level singletons (_session in inference.ts)
    mockReset: false,
    restoreMocks: false,
  },
  resolve: {
    alias: {
      "@": path.resolve(__dirname),
    },
  },
});
