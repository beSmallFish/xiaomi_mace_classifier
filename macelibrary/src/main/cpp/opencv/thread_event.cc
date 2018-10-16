#include <pthread.h>
#include <stdio.h>
#include <unistd.h>
#include <stdlib.h>
#include "thread_event.h"

    void thread_event_close(struct thread_event_t &event){
        //log_error("thread_event close");
       // if(event) {
            pthread_mutex_destroy(&event.pthread_mutex);
            pthread_cond_destroy(&event.pthread_cond);
      //  }
        return ;
    }

    int32_t thread_event_init(struct thread_event_t &event) {
        //event->value = 0;
        event.value = 0;
        int ret = pthread_mutex_init(&event.pthread_mutex, NULL);
        if(ret >= 0) {
            ret = pthread_cond_init(&event.pthread_cond, NULL);
        }
        if(ret < 0){
            //log_error("thread_event init failed!");
            thread_event_close(event);
        }
        return ret;
    }

    int32_t thread_event_wait(struct thread_event_t &event){
        int32_t result = 0;
        pthread_mutex_lock(&event.pthread_mutex);
        event.value += 1;
        if (event.value > 0){
            //log_error("thread_event wait, value -> %d",  event.value);
            result =  pthread_cond_wait(&event.pthread_cond, &event.pthread_mutex);
        }
        pthread_mutex_unlock(&event.pthread_mutex);
        return result;
    }

    int32_t thread_event_signal(struct thread_event_t &event){
        int32_t result = 0;
        pthread_mutex_lock(&event.pthread_mutex);
        if(event.value > 0){
           // log_error("thread_event signal, value <- %d",  event.value);
            result = pthread_cond_signal(&event.pthread_cond);
        }
        event.value -= 1;
        pthread_mutex_unlock(&event.pthread_mutex);
        return result;
    }


