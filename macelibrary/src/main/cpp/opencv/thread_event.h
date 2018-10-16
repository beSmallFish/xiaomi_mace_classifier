#include <pthread.h>
#include <stdio.h>
#include <unistd.h>
#include <stdlib.h>

    struct thread_event_t {
        pthread_mutex_t pthread_mutex;
        pthread_cond_t pthread_cond;
        volatile int32_t value;
    };

    void thread_event_close(struct thread_event_t &event);

    int32_t thread_event_init(struct thread_event_t &event);

    int32_t thread_event_wait(struct thread_event_t &event);

    int32_t thread_event_signal(struct thread_event_t &event);



