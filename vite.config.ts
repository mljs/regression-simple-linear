/// <reference types="vitest" />
import { defineConfig } from 'vitest/config';
import react from '@vitejs/plugin-react';

export default defineConfig({
  test: {
    include: ['**/*.{test,spec}.ts'],
    globals: true,
  },
  plugins: [react()],
});
