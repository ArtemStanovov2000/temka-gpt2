import { configureStore } from "@reduxjs/toolkit";
import contextSlice from "./contextSlice";

export default configureStore({
    reducer: {
        context: contextSlice,
    }
})
