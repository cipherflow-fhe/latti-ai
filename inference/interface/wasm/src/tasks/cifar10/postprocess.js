import {classify} from '../classification.js';
export const postprocess=options=>classify({...options,classes:10});
