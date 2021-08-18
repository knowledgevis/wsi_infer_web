import Vue from 'vue'
import Router from 'vue-router'
import Home from './views/Home.vue'
import infer_wsi from './apps/infer_wsi.vue'


Vue.use(Router)

export default new Router({
  routes: [
    {
      path: '/',
      name: 'home',
      component: Home
    },
     {
      path: '/infer_wsi',
      name: 'infer_wsi',
      component: infer_wsi,
    },

  ]
})
