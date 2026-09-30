// A helper of FlowThreads (flow-threads.js): runs its share of each pass (core/threads.js).
import { serve } from '../core/threads.js';
import { work } from './flow-threads.js';

serve(work);
