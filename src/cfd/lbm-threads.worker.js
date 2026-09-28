// A helper of LBMThreads (lbm-threads.js): runs its share of each step (core/threads.js).
import { serve } from '../core/threads.js';
import { work } from './lbm-threads.js';

serve(work);
