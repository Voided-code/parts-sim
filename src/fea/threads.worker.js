// A helper of Threads (threads.js): runs its share of each call (core/threads.js).
import { serve } from '../core/threads.js';
import { work, onShare } from './threads.js';

serve(work, onShare);
