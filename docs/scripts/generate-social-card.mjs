import path from 'node:path';
import { fileURLToPath } from 'node:url';
import sharp from 'sharp';

const docsRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const source = path.join(docsRoot, 'src/assets/anvil-social-card.svg');
const output = path.join(docsRoot, 'public/anvil-social-card.png');

await sharp(source).png({ compressionLevel: 9, palette: true }).toFile(output);
