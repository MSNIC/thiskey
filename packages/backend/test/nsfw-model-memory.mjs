/*
 * SPDX-FileCopyrightText: syuilo and misskey-project
 * SPDX-License-Identifier: AGPL-3.0-only
 */

import assert from 'node:assert/strict';
import { createRequire } from 'node:module';
import * as tf from '@tensorflow/tfjs';

// Run with TFJS_NATIVE=1 on a supported platform to also exercise Node.js 24 compatibility.
if (process.env.TFJS_NATIVE === '1') await import('@tensorflow/tfjs-node');

const fixture = tf.sequential({ layers: [
	tf.layers.globalAveragePooling2d({ inputShape: [299, 299, 3] }),
	tf.layers.dense({ units: 5, activation: 'softmax' }),
] });
let artifacts;
await fixture.save(tf.io.withSaveHandler(async value => {
	artifacts = value;
	return { modelArtifactsInfo: tf.io.getModelArtifactsInfoForJSON(value) };
}));
fixture.dispose();

const baseline = tf.memory();
function assertMemory(expected) {
	const actual = tf.memory();
	assert.equal(actual.numTensors, expected.numTensors, 'tensor count must not grow');
	assert.equal(actual.numDataBuffers, expected.numDataBuffers, 'data buffers must not grow');
	assert.equal(actual.numBytes, expected.numBytes, 'tensor bytes must not grow');
}

for (const [entry, nsfw] of [
	['ESM', await import('nsfwjs')],
	['CommonJS', createRequire(import.meta.url)('nsfwjs')],
]) {
	const originalPredict = tf.LayersModel.prototype.predict;
	try {
		tf.LayersModel.prototype.predict = () => { throw new Error('forced warm-up failure'); };
		for (let i = 0; i < 10; i++) {
			await assert.rejects(nsfw.load(tf.io.fromMemory(artifacts), { size: 299 }), /forced warm-up failure/);
			assertMemory(baseline);
		}
	} finally {
		tf.LayersModel.prototype.predict = originalPredict;
	}

	const model = await nsfw.load(tf.io.fromMemory(artifacts), { size: 299 });
	const loaded = tf.memory();
	for (let i = 0; i < 5; i++) {
		const image = tf.zeros([299, 299, 3], 'int32');
		try {
			const predictions = await model.classify(image);
			assert.equal(predictions.length, 5);
		} finally {
			image.dispose();
		}
		assertMemory(loaded);
	}
	model.model.dispose();
	model.normalizationOffset.dispose();
	assertMemory(baseline);
	console.log(`${entry}: 10 failed loads and 5 classifications passed without tensor growth`);
}
