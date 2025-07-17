javascript: (async function copyGenerationDataToClipboard () {
    /* Author: https://github.com/farishijazi */
    url = window.location.href;
    imageId = new URL(url).pathname.split('/').pop();
    var url = "https://civitai.com/api/trpc/image.getGenerationData?input=" + encodeURIComponent('{"json":{"id":' + imageId + ',"authed":true}}');
    fetch(url, {
        "method": "GET",
    }).then(r => r.json()).then(function (response) {

        /*
        here's how this works.

        This code is supposed to get the generation data given a CivitAI image URL.

        generation data from API looks like this:

        ```json
        {
            "result": {
                "data": {
                    "json": {
                        "type": "image",
                        "onSite": true,
                        "process": "txt2img",
                        "meta": {
                            "prompt": "(Soft Lighting Photography by Mimoza Veliu and Mario Giacomelli:1.2), side soft light, Dark environment, black over black, highly detailed, masterful painting in the style of Anders Zorn and Aleksi Briclot, oil on canvas, BadDream, , epiCPhoto, <lora:BadDream, epiCPhoto, Hyperrealism style:0.8>, score_9, score_8_up, score_7_up, score_6_up, \nDetailed and realistic painting depicting a rural scene with a dirt path leading to a quaint, thatched-roof cottage. The layout features a central path flanked by lush greenery and trees, some of which are bare, suggesting early spring. The cottage has white walls with wooden beams and a thatched roof, showing signs of age and wear. Two figures are present: a man in a blue shirt and brown pants, and a woman in a red dress, both standing near the cottage. There are also chickens and ducks scattered along the path, adding to the pastoral atmosphere. The sky is partly cloudy, with patches of blue visible. The painting is signed by the artist in the bottom right corner with the date '1918'.",
                            "negativePrompt": "score_5,score_4,worst quality,low quality,bad anatomy,bad hands,missing fingers,fewer digits,blurry,white background,apron,maid apron, neg - animal paws,man,penis, worst quality, extra fingers, missing fingers, poorly rendered hands, mutation, deformed iris, deformed pupils, deformed limbs, missing limbs, amputee, amputated limbs, watermark, logo , text, piercing, big eyes , teeth, cartoon, shallow depth of field, makeup, nude, breast, niple, nswf,",
                            "cfgScale": 3,
                            "steps": 24,
                            "sampler": "Euler",
                            "seed": 3906952444,
                            "civitaiResources": [
                                {
                                    "type": "checkpoint",
                                    "modelVersionId": 563988
                                },
                                {
                                    "type": "embed",
                                    "weight": 1,
                                    "modelVersionId": 77169
                                },
                                {
                                    "type": "embed",
                                    "weight": 1,
                                    "modelVersionId": 220262
                                },
                                {
                                    "type": "lora",
                                    "weight": 0.8,
                                    "modelVersionId": 349880
                                },
                                {
                                    "type": "lora",
                                    "weight": 0.8,
                                    "modelVersionId": 545175
                                },
                                {
                                    "type": "lora",
                                    "weight": 1,
                                    "modelVersionId": 358694
                                },
                                {
                                    "type": "lora",
                                    "weight": 0.8,
                                    "modelVersionId": 678485
                                }
                            ],
                            "Size": "832x1216",
                            "Created Date": "2024-07-29T1605:46.4544993Z",
                            "clipSkip": 2
                        },
                        "resources": [
                            {
                                "imageId": 21884310,
                                "modelVersionId": 77169,
                                "strength": 1,
                                "modelId": 72437,
                                "modelName": "BadDream + UnrealisticDream (Negative Embeddings)",
                                "modelType": "TextualInversion",
                                "versionId": 77169,
                                "versionName": "BadDream v1.0",
                                "baseModel": "SD 1.5"
                            },
                            {
                                "imageId": 21884310,
                                "modelVersionId": 220262,
                                "strength": 1,
                                "modelId": 195911,
                                "modelName": "epiCPhoto",
                                "modelType": "TextualInversion",
                                "versionId": 220262,
                                "versionName": "epiCPhoto",
                                "baseModel": "SD 1.5"
                            },
                            {
                                "imageId": 21884310,
                                "modelVersionId": 349880,
                                "strength": 0.8,
                                "modelId": 185722,
                                "modelName": "Hyperrealism (3D Vibe) Cinematic Style XL + F1D",
                                "modelType": "LORA",
                                "versionId": 349880,
                                "versionName": "Hyperrealism  xl v2",
                                "baseModel": "SDXL 1.0"
                            },
                            {
                                "imageId": 21884310,
                                "modelVersionId": 358694,
                                "strength": 1,
                                "modelId": 278497,
                                "modelName": "Hand Fine Tuning",
                                "modelType": "LORA",
                                "versionId": 358694,
                                "versionName": "SDXL",
                                "baseModel": "SDXL 1.0"
                            },
                            {
                                "imageId": 21884310,
                                "modelVersionId": 545175,
                                "strength": 0.8,
                                "modelId": 490267,
                                "modelName": "PCM sdxl, normalcfg, 8step, converted_fp16",
                                "modelType": "LORA",
                                "versionId": 545175,
                                "versionName": "16step",
                                "baseModel": "SDXL 1.0"
                            },
                            {
                                "imageId": 21884310,
                                "modelVersionId": 563988,
                                "strength": null,
                                "modelId": 119229,
                                "modelName": "ZavyChromaXL",
                                "modelType": "Checkpoint",
                                "versionId": 563988,
                                "versionName": "v8.0",
                                "baseModel": "SDXL 1.0"
                            },
                            {
                                "imageId": 21884310,
                                "modelVersionId": 678485,
                                "strength": 0.8,
                                "modelId": 251417,
                                "modelName": "Midjourney mimic",
                                "modelType": "LORA",
                                "versionId": 678485,
                                "versionName": "v2.0",
                                "baseModel": "SDXL 1.0"
                            }
                        ],
                        "tools": [],
                        "techniques": [],
                        "external": null,
                        "canRemix": true,
                        "remixOfId": null
                    },
                    "meta": {
                        "values": {
                            "resources.5.strength": [
                                "undefined"
                            ],
                            "external": [
                                "undefined"
                            ],
                            "remixOfId": [
                                "undefined"
                            ]
                        }
                    }
                }
            }
        }
        ```

        the expected output should look like this:

        ```
        (Soft Lighting Photography by Mimoza Veliu and Mario Giacomelli:1.2), side soft light, Dark environment, black over black, highly detailed, masterful painting in the style of Anders Zorn and Aleksi Briclot, oil on canvas, BadDream, , epiCPhoto, <lora:BadDream, epiCPhoto, Hyperrealism style:0.8>, score_9, score_8_up, score_7_up, score_6_up, 
        Detailed and realistic painting depicting a rural scene with a dirt path leading to a quaint, thatched-roof cottage. The layout features a central path flanked by lush greenery and trees, some of which are bare, suggesting early spring. The cottage has white walls with wooden beams and a thatched roof, showing signs of age and wear. Two figures are present: a man in a blue shirt and brown pants, and a woman in a red dress, both standing near the cottage. There are also chickens and ducks scattered along the path, adding to the pastoral atmosphere. The sky is partly cloudy, with patches of blue visible. The painting is signed by the artist in the bottom right corner with the date '1918'.
        Negative prompt: score_5,score_4,worst quality,low quality,bad anatomy,bad hands,missing fingers,fewer digits,blurry,white background,apron,maid apron, neg - animal paws,man,penis, worst quality, extra fingers, missing fingers, poorly rendered hands, mutation, deformed iris, deformed pupils, deformed limbs, missing limbs, amputee, amputated limbs, watermark, logo , text, piercing, big eyes , teeth, cartoon, shallow depth of field, makeup, nude, breast, niple, nswf,
        Additional networks: urn:air:sd1:embed:civitai:72437@77169*1!BadDream, urn:air:sd1:embed:civitai:195911@220262*1!epiCPhoto, urn:air:sdxl:lora:civitai:185722@349880*0.8, urn:air:sdxl:lora:civitai:278497@358694*1, urn:air:sdxl:lora:civitai:490267@545175*0.8, urn:air:sdxl:lora:civitai:251417@678485*0.8
        baseModel: SDXL 1.0, Model: urn:air:sdxl:checkpoint:civitai:119229@563988, Cfg scale: 3, Steps: 24, Sampler: Euler, Seed: 3906952444, Size: 832x1216, Created Date: 2024-07-29T1605:46.4544993Z, Clip skip: 2
        ```
        */

        function processBaseModel(baseModel) {
            var baseModel = baseModel.toLowerCase().replace(' ', '').split('.')[0];
            if (baseModel.toLowerCase().includes('flux')) {
                return 'flux1';
            } else if (baseModel.toLowerCase().includes('sd3')) {
                return 'sd3';
            } else if (baseModel.toLowerCase().includes('sd2')) {
                return 'sd2';
            } else if (baseModel.toLowerCase().includes('sd1')) {
                return 'sd1';
            } else {
                return 'sdxl';
            }
        }

        function convertResponseToGenerationData(response) {
            const data = response.result.data.json;
            const meta = data.meta;
            let processedVersionIds = new Set();

            let generationData = '';

            if (meta.prompt) {
                generationData += meta.prompt.trim();
            }

            if (meta.negativePrompt) {
                generationData += '\nNegative prompt: ' + meta.negativePrompt.trim();
            }

            let additionalNetworks = [];

            if (meta.additionalResources && meta.additionalResources.length > 0) {
                meta.additionalResources.forEach(resource => {
                    const strength = resource.strength || 1.0;
                    const strengthClip = resource.strengthClip || 1.0;

                    const versionIdMatch = resource.name.match(/@(\d+)$/);
                    if (versionIdMatch) {
                        processedVersionIds.add(parseInt(versionIdMatch[1]));
                    }

                    let networkString = resource.name + '*' + strength;
                    if (strengthClip !== 1.0) {
                        networkString += '!' + strengthClip;
                    }
                    additionalNetworks.push(networkString);
                });
            }

            if (data.resources && data.resources.length > 0) {
                data.resources.forEach(resource => {
                    if (resource.modelType === 'Checkpoint') return;

                    if (processedVersionIds.has(resource.modelVersionId)) return;

                    const civitaiResource = meta.civitaiResources?.find(cr =>
                        cr.modelVersionId === resource.modelVersionId
                    );

                    let weight = resource.strength;
                    if (civitaiResource && civitaiResource.weight !== undefined) {
                        weight = civitaiResource.weight;
                    }

                    if (weight !== null && weight !== undefined) {
                        const baseModel = processBaseModel(resource.baseModel.toLowerCase());
                        const modelType = civitaiResource.type;
                        const urn = `urn:air:${baseModel}:${modelType}:civitai:${resource.modelId}@${resource.modelVersionId}`;

                        let networkString = urn + '*' + weight;

                        additionalNetworks.push(networkString);
                        processedVersionIds.add(resource.modelVersionId);
                    }
                });
            }

            if (additionalNetworks.length > 0) {
                generationData += '\nAdditional networks: ' + additionalNetworks.join(', ');
            }


            function convertCamelCaseToSentence(str) {
                if (str.includes(' ')) {
                    return str;
                }
                return str
                    .replace(/([a-z])([A-Z])/g, '$1 $2')
                    .toLowerCase()
                    .replace(/^./, str => str.toUpperCase());
            }

            if (!meta.width || !meta.height) {
                if (meta.size) {
                    const sizeMatch = meta.size.match(/(\d+)x(\d+)/);
                    if (sizeMatch) {
                        meta.width = sizeMatch[1];
                        meta.height = sizeMatch[2];
                    }
                }
            }

            let metadataItems = [];

            const checkpointResource = data.resources?.find(r => r.modelType === 'Checkpoint');
            if (checkpointResource) {
                metadataItems.push(`baseModel: ${checkpointResource.baseModel}`);
            }

            if (meta.Model) {
                metadataItems.push(`Model: ${meta.Model}`);
            } else {
                const checkpointFromCivitAI = meta.civitaiResources?.find(r => r.type === 'checkpoint');
                if (checkpointFromCivitAI && checkpointResource) {
                    let baseModel = processBaseModel(checkpointResource.baseModel);
                    const modelUrn = `urn:air:${baseModel}:checkpoint:civitai:${checkpointResource.modelId}@${checkpointFromCivitAI.modelVersionId}`;
                    metadataItems.push(`Model: ${modelUrn}`);
                }
            }

            const excludedKeys = new Set([
                'prompt', 'negativePrompt', 'additionalResources', 'civitaiResources',
                'Model', 'width', 'height', 'hashes'
            ]);

            for (const [key, value] of Object.entries(meta)) {
                if (!excludedKeys.has(key) && value !== null && value !== undefined &&
                    (typeof value === 'string' || typeof value === 'number' || typeof value === 'boolean')) {
                    if (String(value).trim() !== '') {
                        const label = convertCamelCaseToSentence(key);
                        metadataItems.push(`${label}: ${value}`);
                    }
                }
            }


            if (metadataItems.length > 0) {
                generationData += '\n' + metadataItems.join(', ');
            }

            return generationData;
        };
        generationData = convertResponseToGenerationData(response);
        console.log('Generated data string:');
        console.log(generationData);

        navigator.clipboard.writeText(generationData).then(() => {
            console.log('Generation data copied to clipboard successfully!');
            prompt('✅ Info copied to clipboard:', generationData);
        }).catch(err => {
            console.error('Failed to copy to clipboard:', err);
            const textArea = document.createElement('textarea');
            textArea.value = generationData;
            document.body.appendChild(textArea);
            textArea.select();
            document.execCommand('copy');
            document.body.removeChild(textArea);
            console.log('Fallback copy method used. Please copy the text manually:', generationData);
            prompt('🚨 Fallback copy method used. Please copy the text manually 👇:', generationData);
        });
    }).catch(err => {
        console.error('Failed to fetch generation data:', err);
        prompt('🚨 Failed to fetch generation data (this could mean that the image has missing info):', err);
    });
})();
// https://civitai.com/api/trpc/image.getGenerationData?input=%7B%22json%22%3A%7B%22id%22%3A86363416%2C%22authed%22%3Atrue%7D%7D
