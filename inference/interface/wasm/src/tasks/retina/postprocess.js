import {classify} from '../classification.js';
// Native task output order: [DN, NDRD].
export const postprocess=options=>classify({...options,classes:2});

export {postprocess_batch} from './raw-postprocess.js';
